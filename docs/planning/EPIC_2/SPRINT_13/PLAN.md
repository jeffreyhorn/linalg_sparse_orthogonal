# Sprint 13 Plan: Incomplete Cholesky Preconditioner & MINRES Solver

**Sprint Duration:** 14 days
**Goal:** Add the two highest-value iterative solver extensions: incomplete Cholesky (IC(0)) preconditioning for SPD systems and MINRES for symmetric indefinite systems. IC(0) provides a symmetric positive-definite preconditioner that preserves the sparsity pattern of A, complementing the existing ILU(0)/ILUT preconditioners. MINRES fills the gap for symmetric indefinite systems where CG cannot be used and GMRES is unnecessarily general.

**Starting Point:** A sparse linear algebra library with 15 headers, ~120 public API functions, 892 tests across 32 suites, LU/Cholesky/QR/SVD/LDL^T factorization, CG/GMRES iterative solvers with ILU(0)/ILUT preconditioning, block CG/GMRES for multi-RHS, matrix-free CG/GMRES variants, AMD/RCM fill-reducing reordering, Bunch-Kaufman symmetric pivoting, norm-relative tolerance strategy, factored-state validation, and thread-safe norm caching. The preconditioner callback interface (`sparse_precond_fn`) is well-established.

**End State:** `sparse_ic_factor()`, `sparse_ic_solve()`, and `sparse_ic_precond()` in public API providing IC(0) preconditioning for CG. `sparse_solve_minres()` and `sparse_minres_solve_block()` in public API for symmetric (possibly indefinite) systems with preconditioning support. IC(0) benchmarked against ILU(0) on SPD SuiteSparse matrices. MINRES tested on symmetric indefinite KKT/saddle-point systems.

---

## Day 1: API Design & Data Structures

**Theme:** Design the public APIs and internal data structures for IC(0) and MINRES

**Time estimate:** 8 hours

### Tasks
1. Design IC(0) data structures and API in new `include/sparse_ic.h`:
   - Reuse `sparse_ilu_t` for storage (IC(0) produces L such that L*L^T approx A; store L in the `L` field and L^T in the `U` field, matching how ILU stores L/U)
   - `sparse_ic_factor(const SparseMatrix *A, sparse_ilu_t *ic)` — compute IC(0) factorization
   - `sparse_ic_solve(const sparse_ilu_t *ic, const double *b, double *x)` — forward/backward solve with L and L^T
   - `sparse_ic_precond(const void *ctx, idx_t n, const double *r, double *z)` — preconditioner callback compatible with `sparse_precond_fn`
   - `sparse_ic_free(sparse_ilu_t *ic)` — free IC(0) data (alias for `sparse_ilu_free()`)
2. Design MINRES API additions to `include/sparse_iterative.h`:
   - `sparse_solve_minres(const SparseMatrix *A, const double *b, double *x, const sparse_iter_opts_t *opts, sparse_precond_fn precond, const void *precond_ctx, sparse_iter_result_t *result)` — single-RHS MINRES
   - `sparse_minres_solve_block(const SparseMatrix *A, const double *B, double *X, idx_t nrhs, const sparse_iter_opts_t *opts, sparse_precond_fn precond, const void *precond_ctx, sparse_iter_result_t *result)` — multi-RHS MINRES with per-column convergence
   - Reuse existing `sparse_iter_opts_t` and `sparse_iter_result_t` types
3. Write headers with full Doxygen documentation, `@pre` tags, `@note` for symmetry requirements
4. Create stub `src/sparse_ic.c` with function signatures returning `SPARSE_ERR_NULL`
5. Add MINRES stubs to `src/sparse_iterative.c` (or new `src/sparse_minres.c` if cleaner)
6. Add new source files to Makefile (`LIB_SRCS`) and CMakeLists.txt
7. Verify the library builds: `make clean && make` — no errors
8. Run `make format && make lint` — all clean

### Deliverables
- `include/sparse_ic.h` with complete API documentation
- MINRES function declarations in `include/sparse_iterative.h`
- Stub source files (compile, functions return errors)
- Build system updated

### Completion Criteria
- `make clean && make` builds successfully with new files
- Headers are self-documenting with `@pre`, `@note`, return codes
- `make format && make lint` clean

---

## Day 2: IC(0) Symbolic Phase & Sparsity Pattern

**Theme:** Implement the IC(0) symbolic factorization — determine the nonzero pattern of L

**Time estimate:** 10 hours

### Tasks
1. Implement IC(0) entry validation in `sparse_ic_factor()`:
   - NULL checks, shape checks (must be square)
   - Symmetry check via `sparse_is_symmetric()`
   - SPD requirement documented (IC(0) only works on SPD matrices)
   - Compute and cache factor norm via `sparse_norminf_const()`
2. Implement the symbolic phase of IC(0):
   - Extract the lower triangular sparsity pattern of A (including diagonal)
   - IC(0) preserves this exact pattern — no fill-in beyond what A already has
   - Allocate the L factor as a sparse matrix with this pattern
   - Store diagonal indices for fast access during numeric phase
3. Implement `sparse_ic_free()`:
   - Delegate to `sparse_ilu_free()` or free L/U directly
   - Safe on zeroed struct
4. Write initial tests:
   - Factor a 3x3 SPD diagonal matrix — verify L pattern matches lower triangle
   - Symmetry rejection on non-symmetric input
   - Non-square rejection
   - NULL argument handling
5. Run `make format && make lint && make test` — all clean

### Deliverables
- IC(0) entry validation and symbolic phase
- `sparse_ic_free()` implemented
- Basic validation tests in `tests/test_ic.c`

### Completion Criteria
- Symbolic phase correctly extracts lower triangular pattern
- Validation rejects non-symmetric, non-square, and NULL inputs
- `make format && make lint && make test` clean

---

## Day 3: IC(0) Numeric Factorization & Solve

**Theme:** Implement the IC(0) numeric factorization and triangular solve

**Time estimate:** 10 hours

### Tasks
1. Implement the IC(0) numeric factorization:
   - For each column k = 0..n-1:
     - Compute L(k,k) = sqrt(A(k,k) - sum_{j<k} L(k,j)^2)
     - For each i > k where A(i,k) != 0:
       - L(i,k) = (A(i,k) - sum_{j<k} L(i,j)*L(k,j)) / L(k,k)
     - Drop any fill-in entries not in the original sparsity pattern (IC(0) = no fill)
   - Handle negative diagonal: if A(k,k) - sum < 0, return `SPARSE_ERR_NOT_SPD` (breakdown)
   - Use relative tolerance for near-zero diagonal detection
2. Implement `sparse_ic_solve()`:
   - Forward substitution: solve L*y = b (L is lower triangular)
   - Backward substitution: solve L^T*x = y (L^T is upper triangular)
   - No permutation phase (IC(0) does not reorder)
3. Write numeric factorization tests:
   - 3x3 SPD tridiagonal: verify L*L^T matches A at stored positions
   - Diagonal matrix: L = sqrt(diag(A)), solve is trivial
   - 5x5 banded SPD: verify solve residual ||A*x - b|| / ||b|| < 1e-10
   - Identity matrix: L = I
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Complete IC(0) numeric factorization
- Forward/backward substitution solve
- Numeric correctness tests

### Completion Criteria
- IC(0) factorization produces correct L for SPD matrices
- L*L^T matches A at all positions in the sparsity pattern
- Solve produces small residuals on test matrices
- `make format && make lint && make test` clean

---

## Day 4: IC(0) Preconditioner Callback & CG Integration

**Theme:** Wire IC(0) into the iterative solver framework as a CG preconditioner

**Time estimate:** 8 hours

### Tasks
1. Implement `sparse_ic_precond()`:
   - Cast `ctx` to `const sparse_ilu_t *`
   - Call `sparse_ic_solve(ic, r, z)` — apply M^{-1} = (L*L^T)^{-1}
   - Return `SPARSE_OK` on success
   - Compatible with `sparse_precond_fn` typedef
2. Test IC(0) as CG preconditioner:
   - Build a 20x20 banded SPD system
   - Solve with unpreconditioned CG: record iteration count
   - Solve with IC(0)-preconditioned CG: verify fewer iterations
   - Solve with ILU(0)-preconditioned CG: compare iteration count with IC(0)
   - Verify all three produce the same solution (within tolerance)
3. Test on SuiteSparse SPD matrices:
   - bcsstk04: IC(0)-preconditioned CG vs unpreconditioned CG
   - nos4: same comparison
   - Record iteration counts for benchmarking (Day 12)
4. Edge case tests:
   - Diagonal matrix: IC(0) preconditioner is exact inverse
   - Nearly singular SPD: verify IC(0) doesn't break down
   - Zero RHS: verify x = 0 returned
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_ic_precond()` callback implementation
- IC(0)-preconditioned CG tests with iteration count comparison
- SuiteSparse validation

### Completion Criteria
- IC(0)-preconditioned CG converges in fewer iterations than unpreconditioned
- IC(0) and ILU(0) produce equivalent solutions on SPD systems
- SuiteSparse tests pass with relres < 1e-10
- `make format && make lint && make test` clean

---

## Day 5: MINRES Core — Lanczos Recurrence & QR Factorization

**Theme:** Implement the core MINRES algorithm: Lanczos tridiagonalization with implicit QR

**Time estimate:** 12 hours

### Tasks
1. Implement the Lanczos three-term recurrence for symmetric matrices:
   - Given symmetric A and starting vector v1 = b/||b||:
     - w = A*v_k - beta_{k-1}*v_{k-1}
     - alpha_k = v_k^T * w
     - w = w - alpha_k * v_k
     - beta_k = ||w||
     - v_{k+1} = w / beta_k
   - Store only the last two Lanczos vectors (O(n) storage, not O(n*k))
   - Track the tridiagonal matrix entries: alpha (diagonal), beta (sub/superdiagonal)
2. Implement the implicit QR factorization of the growing tridiagonal:
   - Apply Givens rotations to maintain a QR factorization of the Hessenberg-like system
   - At each step k, apply one new Givens rotation to incorporate the new row
   - Maintain the transformed RHS for the least-squares solution
   - The MINRES residual norm is available cheaply from the QR factorization
3. Implement the MINRES solution update:
   - Use the QR factorization to update x at each iteration via short recurrences
   - Only three direction vectors needed (analogous to CG's short recurrence)
   - This avoids storing the full Krylov basis
4. Write basic MINRES tests:
   - 3x3 SPD diagonal: converge in 1-3 iterations
   - 3x3 symmetric indefinite: verify convergence and residual
   - 5x5 tridiagonal symmetric: verify ||A*x - b|| / ||b|| < tol
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Lanczos recurrence implementation
- Implicit QR with Givens rotations
- MINRES solution update via short recurrences
- Basic convergence tests

### Completion Criteria
- MINRES converges on SPD and symmetric indefinite test matrices
- Residual norm monotonically decreases (MINRES guarantee)
- Solution accuracy: relres < 1e-12 on test matrices
- `make format && make lint && make test` clean

---

## Day 6: MINRES Preconditioning & Convergence Control

**Theme:** Add preconditioning support and robust convergence control to MINRES

**Time estimate:** 12 hours

### Tasks
1. Implement preconditioned MINRES:
   - Replace the standard inner product with the M^{-1}-inner product
   - At each Lanczos step: z = M^{-1} * w, then use z for the recurrence
   - Preconditioner must be SPD for MINRES (even if A is indefinite)
   - Accept preconditioner via `sparse_precond_fn` callback (same as CG/GMRES)
2. Implement convergence control:
   - Check relative residual ||r_k|| / ||b|| against tolerance each iteration
   - Maximum iteration count from `sparse_iter_opts_t`
   - Fill `sparse_iter_result_t`: iterations, final residual, converged flag
   - Handle zero RHS: return x = 0 immediately
   - Handle lucky breakdown (beta_k = 0): exact solution found
3. Test unpreconditioned MINRES:
   - Symmetric positive definite 10x10: compare result with CG
   - Symmetric indefinite 10x10 (KKT-type): verify residual
   - Ill-conditioned symmetric: verify convergence with enough iterations
4. Test preconditioned MINRES:
   - IC(0)-preconditioned MINRES on SPD system: compare iteration count with unpreconditioned
   - Diagonal preconditioner on indefinite system: verify convergence improvement
   - Jacobi (diagonal) preconditioner: simple M = diag(A) where |A(i,i)| > 0
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Preconditioned MINRES with callback interface
- Convergence control matching CG/GMRES conventions
- Preconditioned vs unpreconditioned comparison tests

### Completion Criteria
- Preconditioned MINRES converges in fewer iterations than unpreconditioned
- Result struct populated correctly (iterations, residual, converged flag)
- Zero RHS and lucky breakdown handled
- `make format && make lint && make test` clean

---

## Day 7: MINRES Edge Cases & Robustness

**Theme:** Handle edge cases in MINRES and verify numerical robustness

**Time estimate:** 10 hours

### Tasks
1. Edge case handling:
   - Zero-dimension system: return immediately
   - 1x1 system: direct division, no iteration
   - Already-converged initial guess: detect and return in 0 iterations
   - Singular symmetric system: detect stagnation and return with converged=false
   - Maximum iterations reached: return best solution so far
2. Numerical robustness tests:
   - Extreme-scale matrices (1e-35, 1e+35): verify no overflow/underflow in Lanczos
   - Ill-conditioned symmetric indefinite (condition number ~1e12): verify MINRES converges (possibly slowly)
   - Matrix with eigenvalues clustered near zero: test convergence behavior
   - Nearly symmetric matrix: verify MINRES detects non-symmetry or handles gracefully
3. Lanczos breakdown handling:
   - If beta_k = 0, the Krylov subspace is exhausted — solution is exact
   - If beta_k is tiny but nonzero, potential loss of orthogonality — detect and report
   - Test with matrix that produces early Lanczos termination
4. Comparison tests:
   - MINRES vs GMRES on symmetric indefinite: verify both produce same solution
   - MINRES vs direct LDL^T solve: verify agreement on small systems
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Robust edge case handling for MINRES
- Extreme-scale and ill-conditioned tests
- MINRES vs GMRES/LDL^T comparison tests

### Completion Criteria
- All edge cases produce correct results or clean convergence failure
- Extreme-scale tests pass without overflow/underflow
- MINRES and GMRES agree on symmetric indefinite systems
- `make format && make lint && make test` clean

---

## Day 8: Block MINRES — Multi-RHS Framework

**Theme:** Implement block MINRES for solving multiple right-hand sides simultaneously

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_minres_solve_block()`:
   - Accept matrix B (n x nrhs) and output X (n x nrhs) in column-major layout
   - Per-column convergence tracking: each RHS converges independently
   - Once a column converges, skip it in subsequent iterations (or continue cheaply)
   - Consistent API with `sparse_cg_solve_block()` and `sparse_gmres_solve_block()`
2. Implementation strategy:
   - Option A: Independent MINRES for each RHS (simpler, parallelize later)
   - Option B: True block Lanczos with shared Krylov subspace (more complex, better for clustered eigenvalues)
   - Choose Option A for initial implementation — matches block CG/GMRES pattern
   - Track convergence per column: result.converged indicates all columns converged
3. Write block MINRES tests:
   - 10x10 symmetric indefinite with 3 RHS: verify all columns converge
   - Mixed convergence: one easy RHS, one hard RHS — verify per-column tracking
   - Single RHS via block API: verify matches single-RHS MINRES
   - Zero RHS column: verify x column is zero
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_minres_solve_block()` implementation
- Per-column convergence tracking
- Multi-RHS correctness tests

### Completion Criteria
- Block MINRES solves multiple RHS correctly
- Per-column convergence tracked in result struct
- Single-RHS result matches `sparse_solve_minres()`
- `make format && make lint && make test` clean

---

## Day 9: Block MINRES Completion & Preconditioning

**Theme:** Complete block MINRES with preconditioning and thorough testing

**Time estimate:** 10 hours

### Tasks
1. Add preconditioning to block MINRES:
   - Apply preconditioner to each column independently
   - Verify preconditioned block MINRES converges in fewer iterations
2. Performance and correctness tests:
   - Block MINRES with IC(0) preconditioner on SPD system (5 RHS)
   - Block MINRES with diagonal preconditioner on indefinite system (3 RHS)
   - Compare iteration count: block vs sequential single-RHS calls
   - Verify solutions match between block and sequential approaches
3. Stress tests:
   - Large number of RHS (20 columns) on a 50x50 system
   - All-zero RHS matrix: verify X is all zero
   - Single-column B: verify matches single-RHS API
   - nrhs = 0: verify clean return
4. Memory tests:
   - Verify no leaks under ASan for all block MINRES paths
   - Verify cleanup on early convergence
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Preconditioned block MINRES
- Comprehensive multi-RHS test coverage
- Memory-clean under ASan

### Completion Criteria
- Preconditioned block MINRES converges faster than unpreconditioned
- Block and sequential results agree within tolerance
- No memory leaks under ASan
- `make format && make lint && make test` clean

---

## Day 10: IC(0) + MINRES Integration Testing

**Theme:** Test the full IC(0) + MINRES stack on realistic problems

**Time estimate:** 12 hours

### Tasks
1. IC(0)-preconditioned CG on SPD systems:
   - Build 50x50 banded SPD system: compare IC(0) vs ILU(0) preconditioned CG
   - SuiteSparse bcsstk04: IC(0)-preconditioned CG iteration count and residual
   - SuiteSparse nos4: same comparison
   - Verify IC(0) produces fewer or comparable iterations to ILU(0) on SPD systems
2. MINRES on symmetric indefinite systems:
   - KKT system (50x50): unpreconditioned MINRES vs GMRES
   - KKT system (100x100): MINRES with diagonal preconditioner
   - Saddle-point system: MINRES convergence verification
3. Cross-solver consistency:
   - SPD system: CG, MINRES, and GMRES should all produce the same solution
   - Symmetric indefinite: MINRES and GMRES should agree; CG should fail or diverge
   - Direct vs iterative: LDL^T solve vs MINRES — verify agreement
4. LDL^T as MINRES preconditioner:
   - Implement a simple LDL^T preconditioner callback (if useful for testing)
   - Test MINRES with LDL^T preconditioner on a smaller system
   - This is mainly for validation — in practice you'd use the direct solve
5. Run `make format && make lint && make test` — all clean

### Deliverables
- IC(0) vs ILU(0) comparison on SPD systems
- MINRES on KKT and saddle-point systems
- Cross-solver consistency tests

### Completion Criteria
- IC(0) matches or beats ILU(0) iteration count on SPD systems
- MINRES converges on all symmetric indefinite test systems
- Cross-solver solutions agree within tolerance
- `make format && make lint && make test` clean

---

## Day 11: SuiteSparse Validation & Benchmarks

**Theme:** Validate on real-world SuiteSparse matrices and collect benchmark data

**Time estimate:** 12 hours

### Tasks
1. Write Sprint 13 integration test (`tests/test_sprint13_integration.c`):
   - IC(0) factor and solve on SPD systems
   - IC(0)-preconditioned CG convergence test
   - MINRES on symmetric indefinite KKT system
   - MINRES vs GMRES equivalence on symmetric system
   - Block MINRES multi-RHS test
   - Extreme-scale IC(0) and MINRES tests
   - Unfactored IC(0) solve-before-factor detection
2. SuiteSparse validation:
   - bcsstk04: IC(0)-preconditioned CG — record iterations, residual, time
   - nos4: same tests
   - Construct synthetic symmetric indefinite SuiteSparse-scale system for MINRES
3. Benchmark data collection:
   - IC(0) vs ILU(0) on SPD matrices: factor time, solve time, CG iterations
   - MINRES vs GMRES on symmetric indefinite: iterations, time, memory
   - Record data for documentation (Day 12)
4. Add Sprint 13 integration test to Makefile and CMakeLists.txt
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Sprint 13 integration test
- SuiteSparse validation results
- Benchmark data for IC(0) vs ILU(0) and MINRES vs GMRES

### Completion Criteria
- All SuiteSparse tests pass with relres < 1e-10
- Integration test covers all new functionality
- Benchmark data collected
- `make format && make lint && make test` clean

---

## Day 12: Documentation & Example Programs

**Theme:** Document IC(0) and MINRES and create standalone examples

**Time estimate:** 10 hours

### Tasks
1. Update README:
   - Add IC(0) to preconditioner list and feature table
   - Add MINRES to iterative solver list
   - Update solver comparison table with IC(0) vs ILU(0) benchmark data
   - Update API overview with new functions
   - Update test counts
   - Add to project structure (new header, source)
2. Update `docs/algorithm.md`:
   - Add IC(0) algorithm description (no-fill incomplete Cholesky)
   - Add MINRES algorithm description (Lanczos + implicit QR)
   - Document preconditioner requirements (IC(0) must be SPD)
   - Document MINRES symmetry requirement
3. Create example program `examples/example_ic_minres.c`:
   - Demonstrate IC(0) factorization and use as CG preconditioner
   - Demonstrate MINRES on a symmetric indefinite system
   - Show preconditioned MINRES
   - Compare iteration counts with/without preconditioning
4. Add example to build system (Makefile and CMakeLists.txt)
5. Verify all headers have complete `@pre`, `@note`, `@param`, `@return` documentation
6. Run `make format && make lint && make test` — all clean

### Deliverables
- README updated with IC(0) and MINRES documentation
- Algorithm documentation updated
- Working example program
- Complete header documentation

### Completion Criteria
- Example compiles, runs, and produces correct output
- README accurately describes new functionality
- All new public functions have Doxygen documentation
- `make format && make lint && make test` clean

---

## Day 13: Full Regression & Hardening

**Theme:** Full regression testing, sanitizer runs, and final hardening

**Time estimate:** 10 hours

### Tasks
1. Full regression:
   - `make clean && make test` — all tests pass
   - `make sanitize` — ASan/UBSan clean
   - `make bench` — benchmarks run without crashes
   - CMake build: `mkdir build && cd build && cmake .. && cmake --build . && ctest` — all pass
   - Packaging tests: `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`
2. Memory leak checking:
   - Run all IC(0) tests under ASan — verify no leaks
   - Run all MINRES tests under ASan — verify no leaks
   - Test error paths: singular matrix to IC(0), non-symmetric matrix to MINRES
   - Verify cleanup on all failure paths
3. Fix any issues found during regression:
   - Address any compiler warnings on CI (Linux vs macOS differences)
   - Fix any test flakiness
   - Resolve any sanitizer findings
4. Verify CMake and Makefile test counts match
5. Run `make format && make lint && make test` — final clean build

### Deliverables
- Full regression pass
- Clean sanitizer runs
- CMake and Makefile build parity

### Completion Criteria
- All tests pass — 0 failures across all suites
- `make sanitize` clean — 0 findings
- CMake `ctest` passes all tests
- Packaging tests pass
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final documentation, sprint review, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Final metrics collection:
   - Total test count (expected > 892 with new IC(0), MINRES, and integration tests)
   - IC(0)-specific test count
   - MINRES-specific test count
   - Benchmark comparison: IC(0) vs ILU(0) on reference SPD matrices
   - Benchmark comparison: MINRES vs GMRES on reference symmetric indefinite matrices
2. Write `docs/planning/EPIC_2/SPRINT_13/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics
   - Items deferred (if any)
3. Update project plan if any Sprint 13 items were deferred
4. Run `make format && make lint && make test` — final clean build

### Deliverables
- Sprint retrospective document
- Updated metrics
- Clean final build

### Completion Criteria
- All Sprint 13 items complete or explicitly deferred
- Retrospective written with honest assessment
- `make format && make lint && make test` clean
