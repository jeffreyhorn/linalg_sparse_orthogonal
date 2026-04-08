# Sprint 12 Plan: Sparse LDL^T Factorization (Symmetric Indefinite)

**Sprint Duration:** 14 days
**Goal:** Add sparse LDL^T factorization with Bunch-Kaufman symmetric pivoting for symmetric indefinite systems (KKT, saddle-point, constrained optimization). This is the highest-priority missing factorization identified in the Epic 2 project plan.

**Starting Point:** A sparse linear algebra library with 15 headers, ~110 public API functions, 812 tests across 30 suites, LU/Cholesky/QR/SVD factorization, CG/GMRES iterative solvers with ILU(0)/ILUT preconditioning, norm-relative tolerance strategy, factored-state validation, and thread-safe norm caching. The library supports AMD/RCM fill-reducing reordering and has a well-established pattern for factorization APIs (factor → solve → free, with `factored` flag, `factor_norm`, and `sparse_rel_tol()`).

**End State:** `sparse_ldlt_factor()` and `sparse_ldlt_solve()` in public API with Bunch-Kaufman 1x1/2x2 symmetric pivoting, AMD/RCM reordering support, norm-relative tolerance, factored-state integration, tests on KKT and saddle-point systems, and full documentation.

---

## Day 1: API Design & Data Structures

**Theme:** Design the public API and internal data structures for LDL^T factorization

**Time estimate:** 10 hours

### Tasks
1. Design `sparse_ldlt_t` result struct in new `include/sparse_ldlt.h`:
   - `SparseMatrix *L` — unit lower triangular factor
   - `double *D` — diagonal values (length n; for 2x2 blocks, D[k] and D[k+1] hold the diagonal, with off-diagonal stored separately)
   - `double *D_offdiag` — off-diagonal of 2x2 pivot blocks (length n; zero for 1x1 pivots)
   - `int *pivot_size` — pivot block size at step k: 1 or 2 (length n)
   - `idx_t *perm` — symmetric permutation (length n)
   - `idx_t n` — matrix dimension
   - `double factor_norm` — cached ||A||_inf for relative tolerance
2. Design public API functions:
   - `sparse_ldlt_factor(const SparseMatrix *A, sparse_ldlt_t *ldlt)` — basic factorization
   - `sparse_ldlt_factor_opts(const SparseMatrix *A, const sparse_ldlt_opts_t *opts, sparse_ldlt_t *ldlt)` — with reordering options
   - `sparse_ldlt_solve(const sparse_ldlt_t *ldlt, const double *b, double *x)` — solve Ax = b
   - `sparse_ldlt_free(sparse_ldlt_t *ldlt)` — free factorization data
3. Design `sparse_ldlt_opts_t` options struct:
   - `sparse_reorder_t reorder` — fill-reducing reordering (NONE, RCM, AMD)
   - `double tol` — pivot tolerance for singularity detection (0 for default)
4. Write the header file with full Doxygen documentation, `@pre` tags, `@note` tolerance semantics, return codes
5. Create stub `src/sparse_ldlt.c` with function signatures returning `SPARSE_ERR_NULL`
6. Add to Makefile (`LIB_SRCS`) and CMakeLists.txt (`add_library` sources)
7. Verify the library builds: `make clean && make` — no errors

### Deliverables
- `include/sparse_ldlt.h` with complete API documentation
- `src/sparse_ldlt.c` stub (compiles, functions return errors)
- Build system updated

### Completion Criteria
- `make clean && make` builds successfully with new files
- Header is self-documenting with `@pre`, `@note`, return codes
- `make format && make lint` clean

---

## Day 2: Symmetric Permutation & Norm Computation

**Theme:** Implement preprocessing: symmetric permutation application and norm computation

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_ldlt_factor()` entry validation:
   - NULL checks, shape checks (must be square)
   - Symmetry check via `sparse_is_symmetric()`
   - Reject matrices with non-identity permutations (like ILU)
   - Set `factored = 0` on the output struct fields
   - Compute and cache `factor_norm` via `sparse_norminf_const()`
2. Implement symmetric permutation preprocessing:
   - If reordering requested, compute AMD or RCM permutation
   - Apply symmetric permutation P*A*P^T via `sparse_permute()`
   - Store permutation in `ldlt->perm` for solve-time unpermutation
3. Implement `sparse_ldlt_free()`:
   - Free L, D, D_offdiag, pivot_size, perm
   - Zero-initialize all fields (safe to call on zeroed struct)
4. Write initial tests:
   - Factor a 1x1 symmetric matrix — verify no crash (stub returns error, but preprocessing runs)
   - Verify symmetry rejection on non-symmetric input
   - Verify non-identity permutation rejection
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Entry validation and preprocessing in `sparse_ldlt_factor()`
- `sparse_ldlt_free()` implemented
- Basic validation tests

### Completion Criteria
- Symmetry and shape checks work correctly
- AMD/RCM permutation is applied before factorization
- `sparse_ldlt_free()` is safe on zeroed struct
- `make format && make lint && make test` clean

---

## Day 3: Bunch-Kaufman Pivot Selection — 1x1 Pivots

**Theme:** Implement the Bunch-Kaufman pivot search for 1x1 pivots

**Time estimate:** 12 hours

### Tasks
1. Implement the Bunch-Kaufman pivot selection algorithm (Phase 1: 1x1 only):
   - At step k, examine the diagonal A(k,k) and the largest off-diagonal |A(i,k)| for i > k
   - If |A(k,k)| >= alpha * max_offdiag (alpha = (1+sqrt(17))/8 ≈ 0.6404), use 1x1 pivot
   - Record `pivot_size[k] = 1`
   - For now, fall back to 1x1 pivot for all cases (2x2 deferred to Day 4)
2. Implement the symmetric elimination step for a 1x1 pivot:
   - Compute multipliers: L(i,k) = A(i,k) / A(k,k) for i > k
   - Update trailing submatrix: A(i,j) -= L(i,k) * A(k,k) * L(j,k) for i,j > k
   - Store L entries in the output L matrix, D[k] = A(k,k)
   - Use dense column accumulator for fill-in handling (same pattern as Cholesky)
3. Implement drop tolerance for fill-in:
   - Drop L entries with |L(i,k)| < DROP_TOL * |L(k,k)|
   - Use `sparse_rel_tol()` for singularity detection on D[k]
4. Write 1x1 pivot tests:
   - Positive-definite 3x3: verify L*D*L^T ≈ P*A*P^T
   - Diagonal matrix: verify D = diag(A), L = I
   - Identity matrix: trivial case
5. Run `make format && make lint && make test` — all clean

### Deliverables
- 1x1 Bunch-Kaufman pivot selection
- Symmetric elimination loop (1x1 pivots only)
- L*D*L^T reconstruction tests

### Completion Criteria
- 1x1 pivot factorization produces correct L and D for SPD matrices
- Results match Cholesky (up to D being squared Cholesky diagonal)
- `make format && make lint && make test` clean

---

## Day 4: Bunch-Kaufman Pivot Selection — 2x2 Pivots

**Theme:** Extend Bunch-Kaufman to handle 2x2 pivot blocks for indefinite matrices

**Time estimate:** 12 hours

### Tasks
1. Implement the full Bunch-Kaufman pivot decision:
   - If 1x1 pivot criterion fails, find row r with largest |A(r,k)| for r > k
   - Compute sigma_r = max_{i != r} |A(i,r)| (largest off-diagonal in column r)
   - If |A(k,k)| * sigma_r >= alpha * max_offdiag^2, use 1x1 pivot at (k,k)
   - Else if |A(r,r)| >= alpha * sigma_r, use 1x1 pivot at (r,r) after symmetric swap
   - Else use 2x2 pivot with rows/columns k and r
2. Implement symmetric row/column interchange:
   - Swap rows and columns k <-> r in the working matrix
   - Update permutation tracking
   - Maintain symmetry during swaps
3. Implement the 2x2 pivot elimination step:
   - Form the 2x2 pivot block: D_block = [A(k,k) A(k,k+1); A(k+1,k) A(k+1,k+1)]
   - Invert the 2x2 block: compute D_block^{-1}
   - Compute multipliers: [L(i,k) L(i,k+1)] = [A(i,k) A(i,k+1)] * D_block^{-1}
   - Update trailing submatrix symmetrically
   - Store D[k], D[k+1], D_offdiag[k]; set pivot_size[k] = 2, pivot_size[k+1] = 2
   - Advance k by 2 instead of 1
4. Write 2x2 pivot tests:
   - 2x2 indefinite matrix [[0,1],[1,0]]: must use 2x2 pivot
   - 3x3 with zero diagonal: [[0,1,0],[1,2,1],[0,1,0]]
   - Verify D blocks and L factors reconstruct A
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Full Bunch-Kaufman pivot selection (1x1 and 2x2)
- 2x2 pivot elimination with symmetric row/column interchange
- Tests on indefinite matrices

### Completion Criteria
- Zero-diagonal matrices factorize using 2x2 pivots
- L*D*L^T reconstruction matches P*A*P^T for indefinite matrices
- `make format && make lint && make test` clean

---

## Day 5: Bunch-Kaufman Hardening & Edge Cases

**Theme:** Handle edge cases in pivot selection and improve numerical robustness

**Time estimate:** 10 hours

### Tasks
1. Handle singularity detection:
   - 1x1 pivot: reject if |D[k]| < `sparse_rel_tol(factor_norm, DROP_TOL)`
   - 2x2 pivot: reject if 2x2 block is singular (det ≈ 0)
   - Return `SPARSE_ERR_SINGULAR` with clean error recovery (free partial factorization)
2. Handle pivot growth control:
   - Track element growth during elimination
   - Warn or fail if growth exceeds a threshold (e.g., 1e10 * ||A||_inf)
3. Edge case handling:
   - 0x0 matrix: return immediately
   - 1x1 matrix: trivial factorization
   - Already-diagonal matrix: L = I, D = diag
   - All-zero matrix: singular detection
   - Nearly singular 2x2 blocks: fallback to 1x1 with diagonal modification
4. Write edge case tests:
   - Singular matrix: expect `SPARSE_ERR_SINGULAR`
   - 0x0 and 1x1 matrices
   - Matrix requiring mixture of 1x1 and 2x2 pivots
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Robust singularity detection for both pivot types
- Edge case handling
- Error recovery tests

### Completion Criteria
- All edge cases produce correct results or clean errors
- No memory leaks on error paths (verify with ASan)
- `make format && make lint && make test` clean

---

## Day 6: LDL^T Solve — Forward and Backward Substitution

**Theme:** Implement the three-phase solve: L, D, L^T substitution with permutation

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_ldlt_solve()`:
   - Phase 0: Apply permutation — b_perm[i] = b[perm[i]]
   - Phase 1: Forward substitution — solve L*y = b_perm (L is unit lower triangular)
   - Phase 2: Diagonal solve — solve D*z = y
     - For 1x1 pivots: z[k] = y[k] / D[k]
     - For 2x2 pivots: solve the 2x2 system [D[k] D_offdiag[k]; D_offdiag[k] D[k+1]] * [z[k]; z[k+1]] = [y[k]; y[k+1]]
   - Phase 3: Backward substitution — solve L^T*x_perm = z
   - Phase 4: Apply inverse permutation — x[perm[i]] = x_perm[i]
2. Add factored-state check (`ldlt->factored` or check via `factor_norm >= 0`)
3. Write solve tests:
   - SPD 3x3: compare LDL^T solve result with Cholesky solve result
   - Indefinite 3x3: verify A*x = b via residual check
   - Indefinite with 2x2 pivots: verify residual
   - Multiple solves with same factorization
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Complete `sparse_ldlt_solve()` with all four phases
- Solve correctness tests with residual verification

### Completion Criteria
- Solve produces ||A*x - b|| / ||b|| < 1e-12 on test matrices
- SPD results match Cholesky solve
- 2x2 pivot solve is correct
- `make format && make lint && make test` clean

---

## Day 7: Fill-Reducing Reordering Integration

**Theme:** Integrate AMD/RCM reordering with LDL^T and verify fill-in reduction

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_ldlt_factor_opts()`:
   - Apply AMD or RCM symmetric permutation before factorization
   - Pass through to `sparse_ldlt_factor()` after reordering
   - Store combined permutation for solve unpermutation
2. Verify reordering reduces fill-in:
   - Compare nnz(L) with and without AMD on SuiteSparse matrices
   - Verify that solve results are identical regardless of reordering
3. Test on SuiteSparse matrices:
   - bcsstk04 (symmetric positive definite, stiffness matrix)
   - nos4 (symmetric positive definite)
   - Verify factorization succeeds and solve residual is small
4. Write reordering tests:
   - Factor with NONE vs AMD: verify same solve result
   - Factor with RCM: verify same solve result
   - Compare fill-in: nnz(L) with AMD < nnz(L) without AMD on structured matrices
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_ldlt_factor_opts()` with AMD/RCM reordering
- Fill-in comparison tests
- SuiteSparse validation tests

### Completion Criteria
- AMD reduces fill-in on structured matrices
- Solve results identical with/without reordering
- All SuiteSparse tests produce relres < 1e-10
- `make format && make lint && make test` clean

---

## Day 8: KKT & Saddle-Point Matrix Tests

**Theme:** Test LDL^T on the primary use case: KKT and saddle-point systems

**Time estimate:** 10 hours

### Tasks
1. Construct KKT test matrices:
   - Standard form: K = [H A^T; A 0] where H is SPD and A is a constraint matrix
   - Build 2-3 KKT systems of varying size (small 6x6, medium 20x20, larger 50x50)
   - Verify these are symmetric indefinite (have positive and negative eigenvalues)
2. Construct saddle-point test matrices:
   - Stokes-type: [[viscosity block, gradient^T], [gradient, 0]]
   - Verify LDL^T factorizes with 2x2 pivots near the zero block
3. Solve and verify:
   - Build known-solution RHS (b = A * x_exact)
   - Solve with LDL^T and verify ||x - x_exact|| / ||x_exact|| < tolerance
   - Compare with LU solve on same system: verify results agree
4. Write tests:
   - KKT small: verify solve with 2x2 pivots
   - KKT medium: verify residual
   - Saddle-point: verify factorization uses mix of 1x1 and 2x2 pivots
   - Compare LDL^T vs LU: solutions should agree within tolerance
5. Run `make format && make lint && make test` — all clean

### Deliverables
- KKT and saddle-point test matrices
- LDL^T vs LU comparison tests
- Verified solve on indefinite systems

### Completion Criteria
- KKT systems solve with relres < 1e-10
- 2x2 pivots are used where expected (zero diagonal blocks)
- LDL^T and LU produce equivalent solutions
- `make format && make lint && make test` clean

---

## Day 9: Scaled Tolerance & Negative Eigenvalue Tests

**Theme:** Test LDL^T with extreme scales and matrices with known eigenvalue structure

**Time estimate:** 8 hours

### Tasks
1. Scaled matrix tests (using Sprint 11 tolerance infrastructure):
   - Factor and solve at scale 1e-35: verify x ≈ x_exact
   - Factor and solve at scale 1e+35: verify no false singularity
   - Use the `create_scaled_spd` pattern from test_edge_cases.c adapted for indefinite matrices
2. Negative eigenvalue tests:
   - Matrix with known eigenvalues: diag(3, 1, -1, -2) → verify D captures signs
   - Matrix with eigenvalues spanning many magnitudes
   - Verify D diagonal entries have correct signs
3. Inertia computation:
   - Add `sparse_ldlt_inertia()`: count positive, negative, and zero D entries
   - This is a cheap byproduct of factorization — just count signs of D blocks
   - For 2x2 blocks, compute eigenvalues of the 2x2 D block to determine signs
4. Write tests:
   - Inertia of SPD matrix: (n, 0, 0)
   - Inertia of negative-definite matrix: (0, n, 0)
   - Inertia of indefinite matrix: verify matches known eigenvalue count
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Scaled tolerance tests for LDL^T
- `sparse_ldlt_inertia()` function
- Eigenvalue sign / inertia tests

### Completion Criteria
- Scaled tests pass at 1e-35 and 1e+35
- Inertia matches known eigenvalue structure
- `make format && make lint && make test` clean

---

## Day 10: Iterative Refinement & Condition Estimation

**Theme:** Add iterative refinement for LDL^T and condition number estimation

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_ldlt_refine()`:
   - Same pattern as `sparse_lu_refine()`: compute residual, solve correction, update
   - Signature: `sparse_ldlt_refine(A, ldlt, b, x, max_iters, tol)`
   - Reuse factorization for each correction solve
2. Implement `sparse_ldlt_condest()`:
   - Estimate condition number using Hager/Higham 1-norm estimator
   - Requires a transpose solve — for LDL^T, A^T = A, so same factorization works
   - Signature: `sparse_ldlt_condest(A, ldlt, &condest)`
3. Write tests:
   - Ill-conditioned symmetric indefinite: verify refinement improves solution
   - Well-conditioned: verify condest ≈ 1
   - Ill-conditioned: verify condest is large
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_ldlt_refine()` iterative refinement
- `sparse_ldlt_condest()` condition estimation
- Refinement and conditioning tests

### Completion Criteria
- Refinement reduces residual on ill-conditioned systems
- Condition estimate is within 10x of true value on test matrices
- `make format && make lint && make test` clean

---

## Day 11: Documentation & Example Program

**Theme:** Document the LDL^T API and create a standalone example

**Time estimate:** 8 hours

### Tasks
1. Update README:
   - Add LDL^T to feature list
   - Add to solver comparison table
   - Update API overview with new functions
   - Update test counts
   - Add to project structure (new header, source)
2. Update `docs/algorithm.md`:
   - Add LDL^T factorization algorithm description
   - Document Bunch-Kaufman pivot selection
   - Document 2x2 pivot handling in solve
   - Document tolerance strategy (inherits from Sprint 11)
3. Create example program `examples/example_ldlt.c`:
   - Demonstrate LDL^T factorization of a KKT system
   - Show solve, residual check, and inertia computation
   - Show reordering option
4. Add example to build system (Makefile and CMakeLists.txt)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- README updated with LDL^T documentation
- Algorithm documentation
- Working example program

### Completion Criteria
- Example compiles, runs, and produces correct output
- README accurately describes new functionality
- `make format && make lint && make test` clean

---

## Day 12: SuiteSparse Validation & Performance

**Theme:** Validate on real-world SuiteSparse matrices and measure performance

**Time estimate:** 10 hours

### Tasks
1. Test on SuiteSparse matrices:
   - bcsstk04 (symmetric positive definite — verify matches Cholesky result)
   - Try to find/construct a symmetric indefinite SuiteSparse matrix for testing
   - Create synthetic large KKT systems (100x100, 500x500) for performance testing
2. Performance comparison:
   - Compare LDL^T vs LU on symmetric systems: measure factor time and nnz(L)
   - LDL^T should produce ~50% less fill-in than LU on symmetric matrices
   - Measure solve time for both
3. Add SuiteSparse-based test suite:
   - Automated tests using bcsstk04 and nos4
   - Verify residual on each matrix
4. Memory leak checking:
   - Run all LDL^T tests under ASan
   - Verify no leaks on error paths (singular matrices, allocation failures)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- SuiteSparse validation tests
- Performance comparison data (LDL^T vs LU)
- Leak-free under ASan

### Completion Criteria
- All SuiteSparse tests pass with relres < 1e-10
- No memory leaks under ASan
- LDL^T fill-in <= LU fill-in on symmetric matrices
- `make format && make lint && make test` clean

---

## Day 13: Integration Testing & Full Regression

**Theme:** Cross-feature integration tests and full regression

**Time estimate:** 10 hours

### Tasks
1. Write Sprint 12 integration test (`tests/test_sprint12_integration.c`):
   - LDL^T at extreme scales (1e-35, 1e+35)
   - KKT system: factor → solve → refine → verify
   - Reordering: AMD vs none produces identical solutions
   - Inertia computation matches known eigenstructure
   - LDL^T vs LU equivalence on symmetric systems
   - Factored-state flag: unfactored LDL^T → solve fails
2. Full regression:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run without crashes
   - CMake build: `ctest` all pass
   - Packaging tests: `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`
3. Add Sprint 12 test to Makefile and CMakeLists.txt
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Sprint 12 integration test
- Full regression pass
- Clean build under all sanitizers

### Completion Criteria
- All tests pass (updated total count)
- `make sanitize` clean
- CMake and Makefile test counts match
- Packaging tests pass
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final documentation, sprint review, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Final metrics collection:
   - Total test count
   - LDL^T-specific test count
   - Fill-in comparison: LDL^T vs LU on reference matrices
   - Performance comparison data
2. Write `docs/planning/EPIC_2/SPRINT_12/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics
   - Items deferred (if any)
3. Update project plan if any Sprint 12 items were deferred
4. Run `make format && make lint && make test` — final clean build

### Deliverables
- Sprint retrospective document
- Updated metrics
- Clean final build

### Completion Criteria
- All Sprint 12 items complete or explicitly deferred
- Retrospective written with honest assessment
- `make format && make lint && make test` clean
