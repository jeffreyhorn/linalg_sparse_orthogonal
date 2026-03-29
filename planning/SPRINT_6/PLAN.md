# Sprint 6 Plan: Iterative Enhancements & QR

**Sprint Duration:** 14 days
**Goal:** Extend the iterative solver infrastructure with ILUT preconditioning and right preconditioning for GMRES, then implement full-featured sparse QR factorization using Householder reflections. ILUT handles matrices that ILU(0) cannot (e.g., structurally zero diagonals), right preconditioning gives direct access to the true residual, and QR handles rectangular and rank-deficient systems.

**Starting Point:** A sparse linear algebra library with 62 public API functions, 406 tests across 19 test suites, 14 SuiteSparse reference matrices, LU/Cholesky factorization with AMD/RCM reordering, CG/GMRES iterative solvers with ILU(0) preconditioning, parallel SpMV via OpenMP, CI/CD with GitHub Actions, and code quality tooling (clang-format, clang-tidy, cppcheck).

**End State:** `sparse_ilut_factor()` for ILUT preconditioning (handles west0067-type matrices), right preconditioning option for GMRES, column-pivoted sparse QR factorization (`sparse_qr_factor()`), least-squares solver (`sparse_qr_solve()`), rank estimation (`sparse_qr_rank()`), null-space extraction (`sparse_qr_nullspace()`), and fill-reducing column reordering for QR.

---

## Day 1: ILUT — API Design & Core Algorithm

**Theme:** Design the ILUT interface and implement the factorization algorithm

**Time estimate:** 12 hours

### Tasks
1. Design the ILUT API in a new `include/sparse_ilut.h` (or extend `sparse_ilu.h`):
   - `typedef struct { double tol; idx_t max_fill; } sparse_ilut_opts_t`
   - `sparse_err_t sparse_ilut_factor(const SparseMatrix *A, const sparse_ilut_opts_t *opts, sparse_ilu_t *ilu)` — reuse the existing `sparse_ilu_t` struct for L/U storage
   - `sparse_ilut_precond()` callback (same signature as `sparse_ilu_precond`)
   - Default opts: tol = 1e-3, max_fill = 10
2. Implement `sparse_ilut_factor()`:
   - IKJ Gaussian elimination with dual drop rules:
     - Drop rule 1: drop entry if |value| < tol * ||row_i||
     - Drop rule 2: keep at most max_fill largest entries per row in L and U
   - Partial pivoting: swap rows to get the largest diagonal element when current diagonal is small
   - Work on a copy of A (same pattern as ILU(0))
   - Extract L (unit lower triangular) and U (upper triangular with diagonal)
3. Handle edge cases:
   - NULL inputs → SPARSE_ERR_NULL
   - Non-square → SPARSE_ERR_SHAPE
   - Non-identity permutations → SPARSE_ERR_BADARG
   - Zero diagonal after pivoting → SPARSE_ERR_SINGULAR
4. Add `sparse_ilut.c` (or extend `sparse_ilu.c`) to Makefile and CMakeLists.txt
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_ilut_factor()` with threshold dropping and max fill control
- `sparse_ilut_precond()` callback compatible with CG/GMRES
- Build system updated
- Edge-case error handling

### Completion Criteria
- Code compiles with zero warnings
- Existing `make test` passes
- `make format && make lint` clean

---

## Day 2: ILUT — Testing & SuiteSparse Validation

**Theme:** Test ILUT on synthetic and real matrices, compare with ILU(0)

**Time estimate:** 10 hours

### Tasks
1. Write basic ILUT tests:
   - Dense 3×3: ILUT with high fill = exact LU
   - Tridiagonal SPD: ILUT = ILU(0) (no fill to drop)
   - Matrix with fill positions: verify ILUT keeps more fill than ILU(0)
   - west0067: ILUT succeeds where ILU(0) returns SINGULAR (structurally zero diag)
   - NULL inputs, non-square, non-identity perms → proper error codes
2. Test ILUT as preconditioner:
   - ILUT-preconditioned GMRES on west0067 → verify convergence
   - ILUT vs ILU(0) iteration count comparison on steam1, orsirr_1
   - ILUT-preconditioned CG on nos4, bcsstk04
3. Benchmark ILUT quality vs ILU(0): fill-in, iteration reduction, timing
4. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥8 ILUT tests covering factorization, preconditioning, and error handling
- ILUT validated on west0067 (ILU(0) failure case)
- ILUT vs ILU(0) comparison data

### Completion Criteria
- ILUT-GMRES converges on west0067
- ILUT reduces iterations vs ILU(0) on ill-conditioned matrices
- `make format && make lint && make test` clean

---

## Day 3: Right Preconditioning for GMRES

**Theme:** Add right preconditioning option to GMRES

**Time estimate:** 12 hours

### Tasks
1. Extend `sparse_gmres_opts_t` with a `precond_side` field:
   - `typedef enum { SPARSE_PRECOND_LEFT = 0, SPARSE_PRECOND_RIGHT = 1 } sparse_precond_side_t`
   - Default: `SPARSE_PRECOND_LEFT` (backward compatible)
2. Implement right-preconditioned GMRES in `sparse_solve_gmres()`:
   - Arnoldi: w = A * M^{-1} * v_j (apply preconditioner before matvec for right)
   - After solving the Hessenberg system: x = x_0 + M^{-1} * V_k * y_k
   - The GMRES residual norm directly equals ||b - A*x|| (no gap)
3. Write tests:
   - Right-preconditioned GMRES with diagonal preconditioner → same solution as left
   - Right-preconditioned GMRES with ILU(0) on nos4 → verify residual equals reported
   - Right-preconditioned GMRES with ILUT on west0067 → convergence test
   - Right vs left: compare reported residual vs true residual (right should match exactly)
   - Default opts (precond_side=0) → backward compatible with existing left preconditioning
4. Update docs: `sparse_iterative.h` doc comments, `docs/algorithm.md`
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Right preconditioning option in `sparse_gmres_opts_t`
- Right-preconditioned GMRES implementation
- ≥5 tests comparing left vs right preconditioning
- Updated documentation

### Completion Criteria
- Right-preconditioned GMRES residual matches true residual exactly
- All existing left-preconditioning tests still pass
- `make format && make lint && make test` clean

---

## Day 4: Householder Reflections — Design & Core Implementation

**Theme:** Implement Householder vector computation and application

**Time estimate:** 12 hours

### Tasks
1. Design the QR API in a new `include/sparse_qr.h`:
   - `typedef struct { SparseMatrix *R; double *betas; double **v_vectors; idx_t *col_perm; idx_t m, n, rank; } sparse_qr_t`
   - `sparse_err_t sparse_qr_factor(const SparseMatrix *A, sparse_qr_t *qr)`
   - `sparse_err_t sparse_qr_factor_opts(const SparseMatrix *A, const sparse_qr_opts_t *opts, sparse_qr_t *qr)`
   - `void sparse_qr_free(sparse_qr_t *qr)`
2. Implement Householder vector computation:
   - Given dense column vector x of length m, compute v and beta such that (I - beta*v*v^T)*x = ||x||*e_1
   - Use the standard formula: v = x; v[0] += sign(x[0])*||x||; beta = 2/(v^T*v)
   - Handle the zero-vector case (beta = 0)
3. Implement Householder application to a dense column:
   - Apply (I - beta*v*v^T) to a vector y: y = y - beta*v*(v^T*y)
4. Write unit tests for Householder:
   - Known 3-vector → verify reflection zeroes subdiagonal
   - Identity column → trivial reflection
   - Zero vector → no-op (beta = 0)
5. Add `sparse_qr.c` to Makefile and CMakeLists.txt
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr.h` with full API declarations
- Householder vector computation and application (dense)
- ≥3 Householder unit tests
- Build system updated

### Completion Criteria
- Householder reflection correctly zeroes subdiagonal entries
- `make format && make lint && make test` clean

---

## Day 5: Column-Pivoted QR — Core Factorization (Part 1)

**Theme:** Implement the main QR factorization loop with column pivoting

**Time estimate:** 14 hours

### Tasks
1. Implement `sparse_qr_factor()` core loop:
   - Compute initial column norms for pivot selection
   - For k = 0 to min(m,n)-1:
     - Select pivot: column with largest remaining norm
     - Swap columns k and pivot in working matrix and permutation
     - Extract column k below diagonal into dense working vector
     - Compute Householder vector v_k and beta_k
     - Apply Householder reflection to remaining columns k+1..n-1:
       For each column j > k: col_j -= beta_k * v_k * (v_k^T * col_j)
     - Store v_k and beta_k for later Q reconstruction
     - Update column norms (downdate formula: ||col_j||^2 -= col_j[k]^2)
   - Store R as upper triangular sparse matrix
2. Use dense column accumulators for efficiency (sparse columns → dense working buffer → sparse writeback)
3. Handle rectangular matrices (m > n and m < n)
4. Write basic QR tests:
   - 3×3 full-rank matrix → verify R is upper triangular
   - Verify column permutation is valid
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Core QR factorization loop with column pivoting
- R stored as sparse upper triangular matrix
- Householder vectors and betas stored for Q reconstruction
- Basic correctness tests

### Completion Criteria
- QR produces correct R for small test matrices
- R is upper triangular with |R(k,k)| ≥ |R(k+1,k+1)| (pivot ordering)
- `make format && make lint && make test` clean

---

## Day 6: Column-Pivoted QR — Core Factorization (Part 2)

**Theme:** Complete and harden QR factorization, add edge cases

**Time estimate:** 12 hours

### Tasks
1. Handle rank deficiency:
   - Detect when remaining column norms are below tolerance
   - Record numerical rank in `qr->rank`
   - Stop factorization early for rank-deficient matrices
2. Add edge-case tests:
   - Identity matrix → R = I, trivial Householder vectors
   - Rank-deficient matrix (e.g., duplicate columns) → rank < n
   - 1×1 matrix → trivial QR
   - Rectangular tall (5×3) and wide (3×5) matrices
   - Zero matrix → rank 0
   - NULL inputs → SPARSE_ERR_NULL
3. Verify A = Q*R*P^T reconstruction (using Q from Householder vectors):
   - Build Q explicitly for small matrices, multiply Q*R, unpermute columns
   - Verify ||A - Q*R*P^T|| < tolerance
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Rank-deficient matrix handling with early termination
- ≥8 edge-case and reconstruction tests
- `sparse_qr_free()` implemented

### Completion Criteria
- QR correctly identifies rank on rank-deficient matrices
- ||A - Q*R*P^T|| < 1e-10 on all test cases
- `make format && make lint && make test` clean

---

## Day 7: Q Application — Apply Q and Q^T to Vectors

**Theme:** Implement Q and Q^T application without forming Q explicitly

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_qr_apply_q()`:
   - Apply Q to a vector: Q*x = H_1 * H_2 * ... * H_k * x (apply reflectors in order)
   - Apply Q^T to a vector: Q^T*x = H_k * ... * H_2 * H_1 * x (reverse order)
   - Each H_i application: x = x - beta_i * v_i * (v_i^T * x)
   - `typedef enum { SPARSE_QR_Q, SPARSE_QR_QT } sparse_qr_side_t`
2. Write tests:
   - Q*Q^T*x = x (Q is orthogonal)
   - Q^T*Q*x = x
   - Q^T*b for the least-squares setup
   - Rectangular: Q is m×m for m×n matrix (m > n)
3. Implement `sparse_qr_form_q()`:
   - Explicitly form Q as a dense or sparse matrix (for testing/diagnostics)
   - Apply Q to each column of I_m
   - Verify Q^T*Q = I
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_apply_q()` for Q and Q^T application
- `sparse_qr_form_q()` for explicit Q formation
- ≥5 orthogonality and round-trip tests

### Completion Criteria
- Q^T*Q = I verified on test matrices
- Q*Q^T*x = x for all test vectors
- `make format && make lint && make test` clean

---

## Day 8: Least-Squares Solver

**Theme:** Implement QR-based least-squares solver

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_qr_solve()`:
   - Given QR factorization of A (m×n, m ≥ n): solve min ||Ax - b||_2
   - Steps:
     - Apply Q^T to b: c = Q^T * b
     - Extract c_1 = c[0:rank-1] (first `rank` entries)
     - Back-substitute: R[0:rank-1, 0:rank-1] * x_p = c_1
     - Apply column permutation: x[perm[i]] = x_p[i]
   - Return residual norm ||c[rank:]||_2
   - Handle rank deficiency: only solve for rank components, set remaining to 0
2. Write tests:
   - Square full-rank system → QR solve matches LU solve
   - Overdetermined system (5×3) → least-squares solution
   - Known least-squares problem with analytical solution
   - Rank-deficient system → minimum-norm solution
   - Compare with CG/GMRES on square SPD system
3. Test on SuiteSparse matrices:
   - nos4 → QR solve, compare residual with LU/Cholesky
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_solve()` for least-squares problems
- Residual norm computation
- Rank-deficient handling
- ≥6 least-squares tests

### Completion Criteria
- QR solve matches LU solve on square full-rank systems
- Least-squares residual is minimal on overdetermined systems
- `make format && make lint && make test` clean

---

## Day 9: QR Validation on SuiteSparse Matrices

**Theme:** Validate QR on real-world matrices, benchmark, harden

**Time estimate:** 10 hours

### Tasks
1. Test QR on SuiteSparse matrices:
   - nos4 (100×100 SPD) → factor, solve, verify residual
   - bcsstk04 (132×132 SPD) → factor, solve, verify residual
   - west0067 (67×67 unsymmetric) → factor, solve, verify residual
   - Verify ||A - Q*R*P^T|| on each matrix
2. Compare QR solve with LU solve and iterative solvers:
   - Record timing, residual, and fill-in for each method
3. Test QR on rectangular matrices from SuiteSparse (if available) or synthetic:
   - Generate tall (200×50) and wide (50×200) random sparse matrices
   - Verify least-squares and rank estimation
4. Fix any bugs found during real-matrix validation
5. Run `make format && make lint && make test` — all clean

### Deliverables
- QR validated on ≥3 SuiteSparse matrices
- QR vs LU comparison data
- Rectangular matrix tests
- Bugs fixed

### Completion Criteria
- QR residual < 1e-8 on all SuiteSparse test matrices
- ||A - Q*R*P^T|| < 1e-10 on all test cases
- `make format && make lint && make test` clean

---

## Day 10: Rank Estimation & Null Space

**Theme:** Implement numerical rank estimation and null-space extraction

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_qr_rank()`:
   - Estimate numerical rank from R diagonal
   - Tolerance: |R(k,k)| < tol * |R(0,0)| marks the rank boundary
   - Default tol: machine epsilon * max(m,n) * |R(0,0)|
2. Implement `sparse_qr_nullspace()`:
   - Extract null-space basis from trailing columns of Q corresponding to zero R diagonals
   - Return null-space dimension and basis vectors
   - Verify A * null_vector ≈ 0 for each basis vector
3. Write tests:
   - Full-rank matrix → rank = n, empty null space
   - Rank-1 matrix (outer product) → rank = 1, null space dimension = n-1
   - Matrix with known null space → verify extracted vectors
   - Rank-deficient rectangular matrix → verify rank and null space
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_rank()` for numerical rank estimation
- `sparse_qr_nullspace()` for null-space basis extraction
- ≥5 rank and null-space tests

### Completion Criteria
- Rank estimation correct on all test matrices
- A * null_vector ≈ 0 for all extracted null-space vectors
- `make format && make lint && make test` clean

---

## Day 11: QR Integration with Fill-Reducing Reordering

**Theme:** Integrate column reordering to reduce fill-in in R

**Time estimate:** 10 hours

### Tasks
1. Implement column reordering for QR:
   - Compute A^T*A sparsity pattern (without forming the product)
   - Apply AMD reordering to A^T*A pattern to get column permutation
   - Pre-permute columns of A before QR factorization
   - Compose the fill-reducing permutation with QR's column pivoting
2. Add `sparse_qr_opts_t`:
   - `sparse_reorder_t reorder` field (NONE, AMD)
   - `sparse_qr_factor_opts(A, &opts, &qr)` uses reordering
3. Write tests:
   - QR with AMD reordering → same solution as without reordering
   - Compare R fill-in with and without reordering on SuiteSparse matrices
   - Verify solve correctness after reordering
4. Benchmark: R nnz with/without AMD on nos4, west0067, steam1
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Column reordering (AMD-based) for QR
- `sparse_qr_opts_t` with reorder option
- ≥3 reordering integration tests
- Fill-in comparison benchmarks

### Completion Criteria
- Reordered QR produces correct solutions
- Fill-in data shows improvement on at least some matrices
- `make format && make lint && make test` clean

---

## Day 12: Integration Testing & Cross-Feature Validation

**Theme:** Test interactions between all Sprint 6 features

**Time estimate:** 10 hours

### Tasks
1. Cross-feature integration tests:
   - ILUT-preconditioned GMRES (right) on west0067 → convergence
   - QR solve vs ILUT-GMRES on same system → comparable residuals
   - QR on SPD matrix vs Cholesky → same solution
   - Rank estimation on factored matrix matches QR rank
2. Solver comparison tests:
   - Same system solved by LU, Cholesky, CG, GMRES, QR → all produce comparable residuals
3. Edge-case integration:
   - QR on 1×1 system
   - QR solve with zero RHS
   - ILUT with extreme parameters (tol=0 → exact LU, max_fill=0 → heavy dropping)
4. Run full regression:
   - `make test` — all suites pass
   - `make sanitize` — UBSan clean
   - Verify all Sprint 1-5 tests still pass unchanged
5. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥6 cross-feature integration tests
- Solver comparison data
- Full regression pass
- Backward compatibility verified

### Completion Criteria
- All integration tests pass
- All existing tests still pass
- `make format && make lint && make sanitize` clean

---

## Day 13: Documentation & Hardening

**Theme:** Update all documentation, harden edge cases

**Time estimate:** 10 hours

### Tasks
1. Update `docs/algorithm.md`:
   - Add ILUT section (algorithm, drop rules, comparison with ILU(0))
   - Add right preconditioning section (formulation, advantages)
   - Add QR factorization section (Householder, column pivoting, rank estimation)
   - Add least-squares section (QR-based, rank-deficient handling)
2. Update `README.md`:
   - Add ILUT, right preconditioning, QR to feature list
   - Update API overview table (add `sparse_qr.h`, update `sparse_ilu.h`)
   - Update test counts and project structure
   - Add QR usage example
3. Edge-case hardening:
   - Very ill-conditioned matrices → verify QR rank estimation
   - Large fill-in with ILUT → verify memory is bounded by max_fill
   - Singular matrix → QR reports rank < n
4. Run `make format && make lint && make test && make bench` — all clean

### Deliverables
- Updated algorithm documentation
- Updated README
- Edge-case hardening tests
- Clean benchmark runs

### Completion Criteria
- Documentation covers all new features
- All edge cases handled with proper error codes
- `make format && make lint && make test && make bench` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final validation, cleanup, and retrospective

**Time estimate:** 6 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run, no crashes
   - `make format && make lint` — all clean
2. Code review pass:
   - All new public API functions have doc comments
   - All new error codes handled in `sparse_strerror()`
   - `const` correctness on all new functions
   - No compiler warnings with strict flags
3. Verify backward compatibility:
   - Existing code using LU/Cholesky/CG/GMRES/ILU works unchanged
   - No breaking API changes
4. Write `planning/SPRINT_6/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, assertion count, API functions, etc.)
   - ILUT vs ILU(0) comparison data
   - QR vs LU comparison data
   - Items deferred to Sprint 7

### Deliverables
- All tests pass under all sanitizers
- Updated README with new API surface
- Sprint retrospective document
- Clean git history with meaningful commits

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
| 1 | ILUT — API & core algorithm | 12 | `sparse_ilut_factor()`, threshold dropping, partial pivoting |
| 2 | ILUT — testing & validation | 10 | west0067 convergence, ILUT vs ILU(0) comparison, ≥8 tests |
| 3 | Right preconditioning for GMRES | 12 | `precond_side` option, right-preconditioned GMRES, ≥5 tests |
| 4 | Householder — design & implementation | 12 | `sparse_qr.h`, Householder vector/application, ≥3 tests |
| 5 | QR factorization — core (part 1) | 14 | Column-pivoted QR loop, R as sparse upper triangular |
| 6 | QR factorization — core (part 2) | 12 | Rank deficiency, edge cases, A=QRP^T reconstruction, ≥8 tests |
| 7 | Q application | 10 | `sparse_qr_apply_q()`, `sparse_qr_form_q()`, orthogonality tests |
| 8 | Least-squares solver | 12 | `sparse_qr_solve()`, overdetermined/rank-deficient, ≥6 tests |
| 9 | QR SuiteSparse validation | 10 | Real-matrix validation, QR vs LU comparison, rectangular tests |
| 10 | Rank estimation & null space | 10 | `sparse_qr_rank()`, `sparse_qr_nullspace()`, ≥5 tests |
| 11 | QR reordering integration | 10 | AMD column reordering, `sparse_qr_opts_t`, fill-in benchmarks |
| 12 | Integration testing | 10 | Cross-feature tests, solver comparison, full regression |
| 13 | Documentation & hardening | 10 | Algorithm docs, README, edge-case tests |
| 14 | Sprint review & retrospective | 6 | Retrospective, final validation, cleanup |

**Total estimate:** 150 hours (avg ~10.7 hrs/day, max 14 hrs/day)
