# Sprint 8 Plan: Matrix-Free Interface & Sparse SVD

**Sprint Duration:** 14 days
**Goal:** Fix bidiagonal reduction for wide matrices, add a matrix-free interface so iterative solvers can work with implicit operators, then implement full-featured sparse SVD. SVD computes A = U*Σ*V^T and is the most powerful matrix decomposition, enabling rank estimation, pseudoinverse, low-rank approximation, and principal component analysis.

**Starting Point:** A sparse linear algebra library with 86 public API functions, 558 tests across 25 test suites, 14 SuiteSparse reference matrices, LU/Cholesky/QR factorization, CG/GMRES iterative solvers with ILU(0)/ILUT preconditioning (left/right, pivoting), parallel SpMV, economy/sparse-mode QR, QR iterative refinement, dense matrix utilities (Givens, eigensolver), tridiagonal QR eigensolver, Householder bidiagonalization (m ≥ n), sparse transpose, CI/CD with GitHub Actions, and code quality tooling.

**End State:** Bidiagonalization for all rectangular matrices, matrix-free CG/GMRES, full SVD (`sparse_svd()`), truncated SVD (`sparse_svd_partial()`), pseudoinverse (`sparse_pinv()`), low-rank approximation (`sparse_svd_lowrank()`), and SVD-based rank estimation.

---

## Day 1: Bidiagonal Reduction for Wide Matrices

**Theme:** Extend `sparse_bidiag_factor()` to handle m < n via internal transpose

**Time estimate:** 8 hours

### Tasks
1. Modify `sparse_bidiag_factor()` to handle m < n:
   - When m < n, compute A^T internally using `sparse_transpose()`
   - Factor the transpose (n×m, tall): B_t = U_t^T * A^T * V_t
   - Swap U/V in the output: A = V_t * B_t^T * U_t^T
   - Store the transposed bidiagonal (lower bidiagonal → upper bidiagonal)
   - Remove the `SPARSE_ERR_SHAPE` check for m < n
2. Update `sparse_bidiag_free()` if any new fields are needed
3. Write tests:
   - Wide 5×10: reconstruction error ||A - U*B*V^T|| < tol
   - Wide 3×8: verify B dimensions and structure
   - Square (m = n): verify behavior unchanged
   - 1×5 single-row: edge case
4. Update existing wide-matrix test (currently expects SPARSE_ERR_SHAPE)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_bidiag_factor()` works for all m, n (m ≥ n and m < n)
- ≥4 wide-matrix bidiag tests with reconstruction verification
- Existing tall/square tests still pass

### Completion Criteria
- Wide-matrix reconstruction error < 1e-10
- `make format && make lint && make test` clean

---

## Day 2: Matrix-Free Interface — API & CG

**Theme:** Design the matrix-free callback type and implement CG variant

**Time estimate:** 10 hours

### Tasks
1. Define the matrix-free callback type in `include/sparse_iterative.h`:
   - `typedef sparse_err_t (*sparse_matvec_fn)(const void *ctx, idx_t n, const double *x, double *y);`
   - Callback computes y = A*x given context pointer and vector length
2. Implement `sparse_solve_cg_mf()`:
   - Same algorithm as `sparse_solve_cg()` but uses matvec callback instead of `SparseMatrix`
   - Accept `sparse_matvec_fn matvec`, `const void *matvec_ctx`, and dimension `idx_t n`
   - Support same opts, preconditioner, and result structs
3. Write CG matrix-free tests:
   - Wrap a SparseMatrix in a callback and verify CG_mf matches CG
   - Diagonal operator (trivial callback): verify convergence
   - NULL callback → SPARSE_ERR_NULL
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_matvec_fn` callback type
- `sparse_solve_cg_mf()` in public API
- ≥4 matrix-free CG tests

### Completion Criteria
- Matrix-free CG produces same solution as matrix-based CG
- `make format && make lint && make test` clean

---

## Day 3: Matrix-Free Interface — GMRES & Validation

**Theme:** Implement GMRES matrix-free variant and validate on SuiteSparse

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_solve_gmres_mf()`:
   - Same algorithm as `sparse_solve_gmres()` but uses matvec callback
   - Support left and right preconditioning
   - Accept dimension parameters (n for square, or m/n for rectangular if needed)
2. Write GMRES matrix-free tests:
   - Wrap SparseMatrix and verify GMRES_mf matches GMRES
   - Right-preconditioned GMRES_mf on steam1
   - Unsymmetric operator callback
3. Integration tests:
   - Matrix-free CG on nos4 (wrap in callback)
   - Matrix-free GMRES on west0067 with ILUT preconditioner
4. Edge cases:
   - Zero-dimensional system
   - Callback that returns error → propagated to caller
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_solve_gmres_mf()` in public API
- ≥6 matrix-free GMRES tests
- Integration tests with SuiteSparse matrices

### Completion Criteria
- Matrix-free GMRES produces same solution as matrix-based GMRES
- `make format && make lint && make test` clean

---

## Day 4: Golub-Kahan Bidiagonalization — Core

**Theme:** Adapt Householder bidiagonalization into the Golub-Kahan form for SVD

**Time estimate:** 12 hours

### Tasks
1. Create `include/sparse_svd.h` with SVD types:
   - `typedef struct { double *sigma; double *U; double *Vt; idx_t m, n, k; int compute_uv; } sparse_svd_t;`
   - SVD options: `sparse_svd_opts_t` with compute_uv, economy, max_iter, tol
2. Implement Golub-Kahan bidiagonalization:
   - Build on `sparse_bidiag_factor()` from Sprint 7
   - Produce explicit U (m×k) and V (n×k) matrices if requested
   - Apply stored Householder reflectors to form dense U and V
   - Extract bidiagonal B as diagonal + superdiagonal arrays
3. Handle both m ≥ n and m < n (using Day 1's wide-matrix fix)
4. Add to Makefile and CMakeLists.txt
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_svd.h` with SVD types and options
- Golub-Kahan bidiagonalization producing explicit U, V, B
- Build system updated

### Completion Criteria
- ||A - U*B*V^T|| < 1e-10 on small test matrices
- `make format && make lint && make test` clean

---

## Day 5: Golub-Kahan Bidiagonalization — Testing & Validation

**Theme:** Validate Golub-Kahan on rectangular and SuiteSparse matrices

**Time estimate:** 12 hours

### Tasks
1. Test Golub-Kahan on various matrix shapes:
   - Square 5×5: reconstruction and orthogonality of U, V
   - Tall 10×5: verify B is 5×5 bidiagonal
   - Wide 5×10: verify B dimensions correct
   - Rank-deficient matrix: verify handling
2. Test on SuiteSparse matrices:
   - nos4: reconstruction error
   - west0067: unsymmetric case
3. Orthogonality tests:
   - U^T*U ≈ I_k
   - V^T*V ≈ I_k
4. Verify B is truly bidiagonal (zero off-bidiagonal entries)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥8 Golub-Kahan tests
- Validated on SuiteSparse matrices
- Orthogonality verified

### Completion Criteria
- Reconstruction error < 1e-10 on all test cases
- `make format && make lint && make test` clean

---

## Day 6: Implicit QR SVD Iteration — Bulge Chase

**Theme:** Implement the core Golub-Kahan SVD step with Givens rotations

**Time estimate:** 12 hours

### Tasks
1. Implement one implicit QR SVD step on a bidiagonal matrix:
   - Input: bidiagonal B (diagonal + superdiagonal arrays)
   - Compute the implicit shift from the trailing 2×2 of B^T*B
   - Introduce a bulge using initial Givens rotation
   - Chase the bulge down the bidiagonal using alternating left/right Givens rotations
   - Each step zeroes one bulge entry and creates another in the next position
2. Accumulate rotations into U and V:
   - Left rotations update U: U = U * G_left
   - Right rotations update V: V = V * G_right
3. Test single QR step:
   - Verify B remains bidiagonal after the step
   - Verify U*B*V^T ≈ original matrix
   - Verify singular values converge monotonically
4. Run `make format && make lint && make test` — all clean

### Deliverables
- One implicit QR SVD step function
- Givens rotation accumulation into U and V
- ≥4 single-step tests

### Completion Criteria
- Bidiagonal structure preserved after each step
- `make format && make lint && make test` clean

---

## Day 7: Implicit QR SVD Iteration — Shifts & Deflation

**Theme:** Add Wilkinson shift selection and deflation for convergence

**Time estimate:** 12 hours

### Tasks
1. Implement Wilkinson shift selection:
   - Compute eigenvalues of trailing 2×2 of B^T*B using `eigen2x2()`
   - Choose the eigenvalue closer to the last diagonal entry of B^T*B
   - Use as the implicit shift for the QR step
2. Implement deflation:
   - After each QR step, check superdiagonal entries for convergence
   - When |B(k,k+1)| < tol * (|B(k,k)| + |B(k+1,k+1)|), deflate
   - Split bidiagonal into two independent subproblems
   - Handle zero diagonal entries (requires special rotation)
3. Implement the iterative loop:
   - Repeat QR steps until all superdiagonal entries converge
   - Maximum iteration count with NOT_CONVERGED error
   - Track which subblocks are unreduced
4. Test convergence:
   - 3×3 bidiagonal with known singular values
   - Verify convergence rate (should be cubic with Wilkinson shifts)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Wilkinson shift for implicit QR SVD
- Deflation when superdiagonal entries converge
- Iterative loop with convergence check
- ≥4 convergence tests

### Completion Criteria
- Singular values correct to machine precision on test cases
- `make format && make lint && make test` clean

---

## Day 8: Implicit QR SVD Iteration — Testing

**Theme:** Validate SVD iteration on larger matrices and edge cases

**Time estimate:** 10 hours

### Tasks
1. Test on larger bidiagonal matrices:
   - n=20, n=50: verify convergence and singular value accuracy
   - Compare with known singular values (diagonal matrix, rank-1)
2. Test edge cases:
   - Zero superdiagonal entry: should deflate immediately
   - All-zero matrix: singular values all zero
   - Matrix with repeated singular values
   - Very ill-conditioned bidiagonal
3. Verify U*diag(sigma)*V^T ≈ B for the original bidiagonal
4. Performance check: count iterations, verify < 30*n
5. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥6 SVD iteration tests including edge cases
- Iteration count verification
- Reconstruction accuracy verified

### Completion Criteria
- All singular values correct to 1e-12 or better
- `make format && make lint && make test` clean

---

## Day 9: Full SVD Driver — Implementation

**Theme:** Implement `sparse_svd()` top-level driver

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_svd()` in `src/sparse_svd.c`:
   - Input: sparse matrix A, options (compute_uv, economy, tol, max_iter)
   - Steps:
     - Golub-Kahan bidiagonalization: A → U_bidiag * B * V_bidiag^T
     - Implicit QR SVD iteration on B → U_svd * Σ * V_svd^T
     - Compose: U = U_bidiag * U_svd, V = V_bidiag * V_svd
   - Options:
     - `compute_uv = 0`: singular values only (skip U, V composition)
     - `compute_uv = 1` with `economy = 1`: thin U (m×k), V (n×k)
     - `compute_uv = 1` with `economy = 0`: full U (m×m), V (n×n)
   - Sort singular values in descending order (and permute U, V columns accordingly)
2. Implement `sparse_svd_free()` to clean up result
3. Handle convergence failure gracefully
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_svd()` and `sparse_svd_free()` in public API
- Singular values only, thin SVD, and full SVD modes
- Convergence failure handling

### Completion Criteria
- Singular values correct on small test matrices
- `make format && make lint && make test` clean

---

## Day 10: Full SVD Driver — Testing & SuiteSparse Validation

**Theme:** Validate full SVD on real-world matrices

**Time estimate:** 10 hours

### Tasks
1. Test full SVD:
   - Small 3×3: compare singular values with known result
   - Diagonal matrix: singular values = |diagonal entries|
   - Rank-deficient matrix: correct number of zero singular values
   - Rectangular tall (10×5) and wide (5×10)
2. Test on SuiteSparse matrices:
   - nos4 (100×100 SPD): verify ||A - U*Σ*V^T|| < tol
   - west0067 (67×67 unsymmetric): singular values and reconstruction
3. Test options:
   - Singular values only (compute_uv=0): faster, same singular values
   - Economy SVD: U is m×k, V is n×k
4. Compare SVD rank with QR rank on same matrices
5. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥8 full SVD tests
- SuiteSparse validation
- SVD rank matches QR rank

### Completion Criteria
- Reconstruction error < 1e-10 on all test cases
- `make format && make lint && make test` clean

---

## Day 11: Truncated/Partial SVD — Lanczos Bidiagonalization

**Theme:** Implement Lanczos bidiagonalization for computing k largest singular values

**Time estimate:** 12 hours

### Tasks
1. Implement Lanczos bidiagonalization:
   - Iteratively build bidiagonal B of dimension k (user-specified)
   - Generate left vectors p_j and right vectors q_j via:
     - p_{j+1} = A*q_j - alpha_j*p_j (with reorthogonalization)
     - q_{j+1} = A^T*p_{j+1} - beta_{j+1}*q_j (with reorthogonalization)
   - alpha_j and beta_j form the bidiagonal
   - Need A^T*x operation: use `sparse_transpose()` or compute via column traversal
2. Implement `sparse_svd_partial()`:
   - Input: A, k (number of singular values), options
   - Lanczos bidiag for k steps → small k×k bidiagonal
   - Apply SVD iteration to the small bidiagonal
   - Recover approximate singular vectors from Lanczos vectors
3. Handle reorthogonalization for numerical stability
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Lanczos bidiagonalization
- `sparse_svd_partial()` for k largest singular values
- Reorthogonalization for stability

### Completion Criteria
- k largest singular values match full SVD results
- `make format && make lint && make test` clean

---

## Day 12: Truncated/Partial SVD — Testing & Validation

**Theme:** Validate partial SVD accuracy and efficiency

**Time estimate:** 10 hours

### Tasks
1. Test partial SVD:
   - 10×10 matrix, k=3: top 3 singular values match full SVD
   - nos4 (100×100), k=5: top 5 singular values match full SVD
   - Rank-deficient matrix, k > rank: correct handling
   - k = min(m,n): should match full SVD exactly
2. Compare efficiency:
   - Partial SVD with k=5 vs full SVD on nos4: timing comparison
   - Verify partial SVD doesn't compute full bidiagonalization
3. Edge cases:
   - k = 0 → SPARSE_ERR_BADARG
   - k = 1 → single largest singular value
   - Wide matrix (m < n)
4. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥6 partial SVD tests
- Accuracy validation against full SVD
- Edge-case coverage

### Completion Criteria
- Partial SVD top-k singular values match full SVD within 1e-10
- `make format && make lint && make test` clean

---

## Day 13: SVD Applications

**Theme:** Implement pseudoinverse, low-rank approximation, and SVD-based rank estimation

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_svd_rank()`:
   - Count singular values above tolerance
   - Default tol: eps * max(m,n) * sigma_max
   - Return numerical rank
2. Implement `sparse_pinv()`:
   - Pseudoinverse via SVD: A^+ = V * Σ^+ * U^T
   - Σ^+ inverts entries above tolerance, zeros the rest
   - Return as dense matrix (pseudoinverse is generally dense)
   - Verify A * A^+ * A ≈ A
3. Implement `sparse_svd_lowrank()`:
   - Best rank-k approximation: A_k = U_k * Σ_k * V_k^T
   - Use truncated SVD or full SVD with truncation
   - Return as sparse matrix (threshold small entries)
   - Verify ||A - A_k||_F = sqrt(sum sigma_{k+1}^2 + ... + sigma_r^2)
4. Write tests:
   - Rank estimation: full-rank, rank-deficient, nearly singular
   - Pseudoinverse: A*A^+*A ≈ A, A^+*A*A^+ ≈ A^+
   - Low-rank: ||A - A_k|| matches theoretical bound
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_svd_rank()` for numerical rank estimation
- `sparse_pinv()` for pseudoinverse
- `sparse_svd_lowrank()` for best rank-k approximation
- ≥8 SVD application tests

### Completion Criteria
- Pseudoinverse satisfies Moore-Penrose conditions
- Low-rank error matches theoretical bound
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
   - SVD on nos4: compare singular values with eigenvalues of A^T*A
   - Matrix-free GMRES with ILU preconditioner on steam1
   - Partial SVD top-k matches full SVD top-k on bcsstk04
   - SVD rank matches QR rank on rank-deficient matrix
   - Pseudoinverse verify: A * pinv(A) * A ≈ A
3. Code review pass:
   - All new public API functions have doc comments
   - `const` correctness on all new functions
   - No compiler warnings with strict flags
4. Verify backward compatibility:
   - Existing code using all previous APIs works unchanged
   - No breaking API changes
5. Update documentation:
   - Update `docs/algorithm.md` with SVD algorithm description
   - Update `README.md` with new API functions and test counts
6. Write `planning/SPRINT_8/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, API functions, etc.)
   - Items deferred to Sprint 9

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
| 1 | Wide bidiag fix | 8 | `sparse_bidiag_factor()` for m < n, ≥4 tests |
| 2 | Matrix-free CG | 10 | `sparse_matvec_fn`, `sparse_solve_cg_mf()`, ≥4 tests |
| 3 | Matrix-free GMRES & validation | 10 | `sparse_solve_gmres_mf()`, ≥6 tests, SuiteSparse |
| 4 | Golub-Kahan bidiag — core | 12 | `sparse_svd.h`, explicit U/V extraction |
| 5 | Golub-Kahan bidiag — testing | 12 | ≥8 tests, SuiteSparse, orthogonality |
| 6 | Implicit QR SVD — bulge chase | 12 | One SVD step, Givens accumulation, ≥4 tests |
| 7 | Implicit QR SVD — shifts & deflation | 12 | Wilkinson shift, deflation, iterative loop |
| 8 | Implicit QR SVD — testing | 10 | ≥6 tests, edge cases, iteration count |
| 9 | Full SVD driver — implementation | 12 | `sparse_svd()`, `sparse_svd_free()` |
| 10 | Full SVD driver — testing | 10 | ≥8 tests, SuiteSparse, options |
| 11 | Truncated SVD — Lanczos bidiag | 12 | Lanczos bidiag, `sparse_svd_partial()` |
| 12 | Truncated SVD — testing | 10 | ≥6 tests, accuracy vs full SVD |
| 13 | SVD applications | 12 | `sparse_svd_rank()`, `sparse_pinv()`, `sparse_svd_lowrank()`, ≥8 tests |
| 14 | Sprint review & retrospective | 8 | Integration tests, docs, retrospective |

**Total estimate:** 150 hours (avg ~10.7 hrs/day, max 12 hrs/day)
