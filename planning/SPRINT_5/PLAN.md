# Sprint 5 Plan: Iterative Solvers & Preconditioning

**Sprint Duration:** 14 days
**Goal:** Implement Krylov subspace iterative solvers (CG, GMRES) with ILU and Cholesky preconditioning, and add parallel SpMV. These solvers handle larger systems where direct methods are too expensive, and form the iterative backbone needed for QR and SVD convergence loops.

**Starting Point:** A sparse linear algebra library with 56 public API functions, 305 tests across 15 test suites, 14 SuiteSparse reference matrices, Cholesky and LU factorization with AMD/RCM reordering, condition estimation, thread-safe design, CSR/CSC export/import, and sparse matrix-matrix multiply. The library is single-threaded (with optional mutex) and has no iterative solvers.

**End State:** `sparse_solve_cg()` for SPD systems and `sparse_solve_gmres()` for general systems, both with optional preconditioning via `sparse_ilu_factor()`/`sparse_ilu_solve()` or Cholesky. Parallel SpMV via OpenMP gated behind `-DSPARSE_OPENMP`. Convergence benchmarks demonstrating preconditioned vs unpreconditioned performance on SuiteSparse matrices.

---

## Day 1: Conjugate Gradient — API Design & Infrastructure

**Theme:** Design the CG interface and build iterative solver infrastructure

**Time estimate:** 8 hours

### Tasks
1. Design the iterative solver API in a new `include/sparse_iterative.h`:
   - `typedef struct { idx_t max_iter; double tol; int verbose; } sparse_iter_opts_t`
   - `typedef struct { idx_t iterations; double residual_norm; int converged; } sparse_iter_result_t`
   - `typedef sparse_err_t (*sparse_precond_fn)(const void *ctx, const double *r, double *z)` — preconditioner callback: solves M*z = r
   - `sparse_err_t sparse_solve_cg(const SparseMatrix *A, const double *b, double *x, const sparse_iter_opts_t *opts, sparse_precond_fn precond, const void *precond_ctx, sparse_iter_result_t *result)`
   - Default opts: max_iter = 1000, tol = 1e-10, verbose = 0
2. Add `SPARSE_ERR_NOT_CONVERGED` error code to `sparse_err_t` enum in `include/sparse_types.h`
3. Update `sparse_strerror()` to handle the new error code
4. Add stub implementations in a new `src/sparse_iterative.c` that compile cleanly
5. Add `sparse_iterative.c` to Makefile and CMakeLists.txt
6. Design test infrastructure for iterative solvers:
   - Helper to build known SPD test matrices (tridiagonal, Laplacian stencils)
   - Helper to compute residual norm ||b - A*x|| / ||b||

### Deliverables
- `sparse_iterative.h` with full API declarations and doc comments
- `sparse_iter_opts_t`, `sparse_iter_result_t`, `sparse_precond_fn` types
- `SPARSE_ERR_NOT_CONVERGED` error code
- Stub implementations that compile cleanly
- Build system updated
- Test helpers for iterative solver validation

### Completion Criteria
- Code compiles with zero warnings
- Existing `make test` passes

---

## Day 2: Conjugate Gradient — Core Algorithm

**Theme:** Implement the preconditioned CG algorithm

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_solve_cg()` in `src/sparse_iterative.c`:
   - Standard preconditioned CG algorithm:
     - r_0 = b - A*x_0 (initial residual, x_0 = input x as initial guess)
     - If preconditioner: z_0 = M^{-1}*r_0, else z_0 = r_0
     - p_0 = z_0
     - For k = 0, 1, ..., max_iter-1:
       - alpha_k = (r_k^T * z_k) / (p_k^T * A*p_k)
       - x_{k+1} = x_k + alpha_k * p_k
       - r_{k+1} = r_k - alpha_k * A*p_k
       - Check convergence: ||r_{k+1}|| / ||b|| < tol
       - If preconditioner: z_{k+1} = M^{-1}*r_{k+1}, else z_{k+1} = r_{k+1}
       - beta_k = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k)
       - p_{k+1} = z_{k+1} + beta_k * p_k
   - Use `sparse_matvec()` for A*p (already exists in API)
   - Allocate workspace: r, z, p, Ap vectors (4 × n doubles)
   - Return `SPARSE_ERR_NOT_CONVERGED` if max_iter exceeded
   - Populate `sparse_iter_result_t` with iteration count, final residual, converged flag
2. Handle edge cases:
   - Zero RHS (b = 0) → x = 0, zero iterations
   - 1×1 system → direct division
   - Non-square matrix → `SPARSE_ERR_SHAPE`
   - NULL inputs → `SPARSE_ERR_NULL`
3. Write basic CG tests:
   - 2×2 SPD: [[4, 1], [1, 3]] with known solution
   - 3×3 SPD tridiagonal → verify converges in ≤ n iterations (exact arithmetic property)
   - Identity matrix → x = b in 1 iteration
   - Zero RHS → x = 0

### Deliverables
- Working `sparse_solve_cg()` for SPD systems
- ≥4 basic CG tests

### Completion Criteria
- CG converges to correct solution on known SPD test cases
- Relative residual < tol on all passing tests
- `make test` passes

---

## Day 3: Conjugate Gradient — Validation & Convergence Testing

**Theme:** Validate CG on larger matrices and test convergence behavior

**Time estimate:** 9 hours

### Tasks
1. Test CG on SPD SuiteSparse matrices:
   - nos4 (100×100) — solve with random RHS, verify residual < tol
   - bcsstk04 (132×132) — solve with random RHS, verify residual < tol
   - Compare iteration count with and without initial guess (x_0 = 0 vs x_0 = direct solve approximation)
2. Test convergence monitoring:
   - Verify iteration count is populated correctly in result struct
   - Verify residual_norm decreases monotonically (CG property on SPD)
   - Test with very tight tolerance (1e-14) → may need more iterations
   - Test with loose tolerance (1e-4) → fewer iterations
3. Test non-SPD rejection:
   - Non-symmetric matrix → CG may not converge or produce garbage (document behavior: CG is for SPD only, user responsibility)
   - Indefinite symmetric matrix → CG breakdown (alpha denominator can be zero/negative)
4. Compare CG result with direct solve (Cholesky) on same systems:
   - Verify CG and Cholesky produce solutions with comparable residuals
5. Add verbose mode test: verify iteration log output when verbose=1

### Deliverables
- CG validated on nos4 and bcsstk04
- Convergence behavior tested with various tolerances
- CG vs Cholesky comparison on same systems
- ≥6 additional CG tests

### Completion Criteria
- CG residual < 1e-10 on nos4 and bcsstk04
- CG solution matches Cholesky solution within tolerance
- `make test` and `make sanitize` clean

---

## Day 4: GMRES — Core Algorithm (Part 1)

**Theme:** Implement the Arnoldi process and GMRES foundation

**Time estimate:** 10 hours

### Tasks
1. Add GMRES declaration to `include/sparse_iterative.h`:
   - `sparse_err_t sparse_solve_gmres(const SparseMatrix *A, const double *b, double *x, const sparse_gmres_opts_t *opts, sparse_precond_fn precond, const void *precond_ctx, sparse_iter_result_t *result)`
   - `typedef struct { idx_t max_iter; idx_t restart; double tol; int verbose; } sparse_gmres_opts_t`
   - Default opts: max_iter = 1000, restart = 30, tol = 1e-10, verbose = 0
2. Implement the Arnoldi process (core of GMRES):
   - Build an orthonormal basis V = [v_1, ..., v_k] for the Krylov subspace
   - Compute the upper Hessenberg matrix H_k via modified Gram-Schmidt
   - V stored as a flat array of column vectors (k × n)
   - H stored as a flat (k+1) × k upper Hessenberg matrix
3. Implement Givens rotations for solving the Hessenberg least-squares problem:
   - Apply Givens rotations to H and the residual vector incrementally
   - This avoids forming and solving the full least-squares problem each iteration
   - Track the residual norm from the Givens rotation machinery
4. Implement the basic (non-restarted) GMRES:
   - Run Arnoldi for up to restart steps
   - Solve the triangular system from Givens-transformed H
   - Form the solution: x = x_0 + V_k * y_k
5. Write basic GMRES tests:
   - 2×2 general (non-symmetric) system → converges in ≤ 2 iterations
   - 3×3 system → verify solution
   - Identity → x = b in 1 iteration

### Deliverables
- Arnoldi process implementation
- Givens rotation least-squares solver
- Basic (non-restarted) GMRES working on small systems
- ≥3 basic GMRES tests

### Completion Criteria
- GMRES converges on small non-symmetric systems
- Arnoldi vectors are orthonormal (verify V^T * V ≈ I)
- `make test` passes

---

## Day 5: GMRES — Restart & Left Preconditioning

**Theme:** Add GMRES restart and preconditioner support

**Time estimate:** 10 hours

### Tasks
1. Implement restarted GMRES(k):
   - After k Arnoldi steps without convergence, update x and restart
   - Outer loop: up to max_iter/restart restarts
   - Recompute r = b - A*x at each restart
   - Track total iteration count across restarts
2. Implement left preconditioning:
   - Instead of solving A*x = b, solve M^{-1}*A*x = M^{-1}*b
   - Apply preconditioner to each Arnoldi vector: v = M^{-1}*(A*v_j) instead of v = A*v_j
   - Preconditioner is the same callback type `sparse_precond_fn`
3. Handle edge cases:
   - Zero RHS → x = 0
   - Non-square → `SPARSE_ERR_SHAPE`
   - restart > n → effectively unrestarted GMRES
   - Breakdown: Arnoldi produces zero vector (lucky breakdown → exact solution found)
4. Write GMRES tests:
   - Unsymmetric 4×4 system → GMRES converges, CG would not
   - Restarted GMRES: test with restart=5 on a 20×20 system → more outer iterations than restart=20
   - Lucky breakdown test: solve a system that lies in a small Krylov subspace
   - Zero RHS, NULL inputs, non-square → proper error handling

### Deliverables
- Restarted GMRES(k) implementation
- Left preconditioning support
- ≥5 GMRES tests including restart behavior

### Completion Criteria
- GMRES(k) converges on unsymmetric systems
- Restart reduces memory usage (only k Arnoldi vectors stored)
- `make test` and `make sanitize` clean

---

## Day 6: GMRES — SuiteSparse Validation

**Theme:** Validate GMRES on real unsymmetric matrices

**Time estimate:** 8 hours

### Tasks
1. Test GMRES on unsymmetric SuiteSparse matrices:
   - west0067 (67×67) — solve with random RHS, verify residual
   - steam1 (240×240) — solve, verify residual
   - orsirr_1 (1030×1030) — solve, verify residual (larger system, good stress test)
2. Compare GMRES convergence with different restart values:
   - GMRES(10), GMRES(30), GMRES(50) on each matrix
   - Record iteration count and residual for each
   - Identify which restart value works best
3. Compare GMRES with direct LU solve:
   - Verify GMRES and LU produce solutions with comparable residuals
   - Record timing: GMRES iterations vs LU factor+solve
4. Test GMRES on SPD matrices (should work, just slower than CG):
   - nos4 with GMRES → compare iteration count with CG
5. Fix any bugs found during real-matrix validation

### Deliverables
- GMRES validated on west0067, steam1, orsirr_1
- Restart comparison data
- GMRES vs LU comparison
- ≥4 SuiteSparse validation tests

### Completion Criteria
- GMRES residual < 1e-8 on all SuiteSparse test matrices
- GMRES solution matches LU solution within tolerance
- `make test` and `make sanitize` clean

---

## Day 7: ILU(0) Preconditioner — Implementation

**Theme:** Implement ILU(0) factorization and solve

**Time estimate:** 10 hours

### Tasks
1. Add ILU declarations to a new `include/sparse_ilu.h`:
   - `typedef struct { SparseMatrix *L; SparseMatrix *U; } sparse_ilu_t`
   - `sparse_err_t sparse_ilu_factor(const SparseMatrix *A, sparse_ilu_t *ilu)` — ILU(0) factorization
   - `sparse_err_t sparse_ilu_solve(const sparse_ilu_t *ilu, const double *r, double *z)` — apply preconditioner: solve L*U*z = r
   - `void sparse_ilu_free(sparse_ilu_t *ilu)`
2. Implement `sparse_ilu_factor()`:
   - ILU(0): compute L and U factors but only allow nonzeros in positions where A has nonzeros
   - Algorithm: IKJ variant of Gaussian elimination
     - For each row i = 1..n-1:
       - For each k < i where A(i,k) != 0:
         - A(i,k) = A(i,k) / A(k,k)
         - For each j > k where A(k,j) != 0:
           - If A(i,j) exists in sparsity pattern: A(i,j) -= A(i,k) * A(k,j)
           - Else: drop (this is the ILU(0) "no fill" rule)
   - Work on a copy of A to preserve the original
   - Extract L (lower triangle with unit diagonal) and U (upper triangle with diagonal)
3. Implement `sparse_ilu_solve()`:
   - Forward substitution with L: solve L*y = r
   - Backward substitution with U: solve U*z = y
4. Write basic ILU tests:
   - 3×3 dense matrix: ILU(0) = exact LU (no fill dropped)
   - Diagonal matrix: ILU = identity factorization
   - Known matrix with fill positions: verify ILU drops fill correctly
   - NULL inputs → proper error codes

### Deliverables
- `sparse_ilu.h` with full API declarations
- `sparse_ilu_factor()` implementing ILU(0)
- `sparse_ilu_solve()` for applying the preconditioner
- `sparse_ilu_free()`
- ≥4 basic ILU tests

### Completion Criteria
- ILU(0) produces correct L and U for test cases
- ILU(0) on dense matrix matches exact LU
- Apply ILU solve → verify L*U*z ≈ r
- `make test` passes

---

## Day 8: ILU(0) — Preconditioner Integration & Validation

**Theme:** Integrate ILU with CG and GMRES, validate on real matrices

**Time estimate:** 9 hours

### Tasks
1. Create preconditioner wrapper for ILU:
   - Implement a `sparse_precond_fn`-compatible wrapper that calls `sparse_ilu_solve()`
   - Pass `sparse_ilu_t *` as the `precond_ctx` void pointer
2. Test ILU-preconditioned CG on SPD matrices:
   - nos4: compare iteration count — unpreconditioned CG vs ILU-preconditioned CG
   - bcsstk04: same comparison
   - Verify both produce correct solutions
3. Test ILU-preconditioned GMRES on unsymmetric matrices:
   - west0067: unpreconditioned vs ILU-preconditioned iteration count
   - steam1: same comparison
   - orsirr_1: same comparison
4. Create Cholesky preconditioner wrapper:
   - Use Cholesky factor of a related SPD matrix (e.g., A^T*A or diagonal) as preconditioner for CG
   - Test on nos4
5. Validate preconditioner quality:
   - ILU should reduce iteration count significantly (≥2× fewer iterations)
   - Preconditioned residual should still meet tolerance

### Deliverables
- ILU preconditioner wrapper compatible with iterative solver API
- ILU-preconditioned CG tests on SPD SuiteSparse matrices
- ILU-preconditioned GMRES tests on unsymmetric SuiteSparse matrices
- Cholesky preconditioner wrapper
- ≥6 preconditioner integration tests

### Completion Criteria
- ILU preconditioning reduces iteration count on all test matrices
- Preconditioned solvers produce correct solutions
- `make test` and `make sanitize` clean

---

## Day 9: Parallel SpMV — OpenMP Implementation

**Theme:** Add OpenMP parallelization to sparse matrix-vector multiply

**Time estimate:** 9 hours

### Tasks
1. Add OpenMP support to build system:
   - Add `-DSPARSE_OPENMP` compile-time flag
   - Add `-fopenmp` to CFLAGS when `SPARSE_OPENMP` is defined
   - Update Makefile with `omp` target: `make omp` builds with OpenMP enabled
   - Update CMakeLists.txt with optional OpenMP support
2. Implement parallel SpMV in `sparse_matvec()`:
   - Guard with `#ifdef SPARSE_OPENMP`
   - Row-wise partitioning: `#pragma omp parallel for schedule(dynamic, 64)`
   - Each thread computes a subset of output rows independently
   - No synchronization needed (each row writes to its own y[i])
   - Dynamic scheduling to handle load imbalance from varying row lengths
3. Verify correctness:
   - Parallel SpMV produces identical results to serial SpMV
   - Test on all SuiteSparse matrices: compare parallel vs serial output
   - Test with 1, 2, 4 threads (via `OMP_NUM_THREADS`)
4. Run under Thread Sanitizer:
   - Compile with `-fsanitize=thread` and `-fopenmp`
   - Verify no data races in parallel SpMV
5. Handle edge cases:
   - Matrix with 0 rows → no-op
   - Matrix with 1 row → single-threaded execution
   - n < num_threads → some threads idle (should be fine with dynamic scheduling)

### Deliverables
- OpenMP-enabled SpMV with compile-time flag
- Build system support for OpenMP
- Correctness verification (parallel == serial)
- TSan-clean parallel SpMV
- ≥4 parallel SpMV tests

### Completion Criteria
- Parallel SpMV produces bit-identical results to serial
- TSan reports zero data races
- Build works with and without `-DSPARSE_OPENMP`
- `make test` clean (both serial and OpenMP builds)

---

## Day 10: Parallel SpMV — Benchmarking & Optimization

**Theme:** Benchmark parallel SpMV speedup and optimize scheduling

**Time estimate:** 8 hours

### Tasks
1. Add SpMV benchmark to `bench_main.c`:
   - `--spmv` flag to benchmark SpMV separately
   - Report throughput: GFLOP/s (2 × nnz operations per SpMV)
   - Report timing for 1000 SpMV iterations (amortize overhead)
   - Compare serial vs parallel (1, 2, 4, 8 threads)
2. Benchmark on SuiteSparse matrices:
   - nos4 (100×100, 594 nnz) — small, likely communication-bound
   - bcsstk04 (132×132, 3648 nnz) — small-medium
   - steam1 (240×240) — medium
   - orsirr_1 (1030×1030) — largest, best candidate for speedup
3. Experiment with scheduling strategies:
   - `static` vs `dynamic` vs `guided` scheduling
   - Vary chunk sizes for dynamic scheduling
   - Determine best default scheduling policy
4. Test parallel SpMV with iterative solvers:
   - Run CG with OpenMP-enabled SpMV → verify same convergence
   - Run GMRES with OpenMP-enabled SpMV → verify same convergence
   - Measure total solve time improvement from parallel SpMV
5. Document performance findings

### Deliverables
- SpMV benchmarks with thread scaling data
- Scheduling strategy comparison
- Parallel SpMV integrated with iterative solvers
- Performance documentation

### Completion Criteria
- Benchmark runs without errors
- Parallel SpMV shows speedup on orsirr_1 with ≥2 threads
- Iterative solvers produce correct results with parallel SpMV
- `make test` clean

---

## Day 11: Integration Testing & Cross-Feature Validation

**Theme:** Test interactions between all Sprint 5 features

**Time estimate:** 8 hours

### Tasks
1. Cross-feature integration tests:
   - CG with ILU preconditioning on all SPD SuiteSparse matrices → verify correct solve
   - GMRES with ILU preconditioning on all unsymmetric SuiteSparse matrices → verify correct solve
   - CG with Cholesky preconditioning → verify faster convergence than ILU on SPD
   - GMRES with restart=10 + ILU on orsirr_1 → convergence test
2. Solver comparison tests:
   - Same system solved by CG, GMRES, LU, and Cholesky → all produce comparable residuals
   - Record iteration counts and timing for each solver
3. Edge-case integration:
   - Iterative solver on a 1×1 system
   - Iterative solver with zero tolerance → runs to max_iter
   - Preconditioner that is the identity (passthrough) → same as unpreconditioned
4. Run full regression:
   - `make test` — all suites pass
   - `make sanitize` — UBSan clean
   - Verify all Sprint 1-4 tests still pass unchanged
5. Verify backward compatibility:
   - Existing API unchanged (no breaking changes)
   - New headers are purely additive

### Deliverables
- ≥6 cross-feature integration tests
- Solver comparison data (CG vs GMRES vs direct)
- Full regression pass
- Backward compatibility verified

### Completion Criteria
- All integration tests pass
- All 305+ existing tests still pass
- `make sanitize` clean

---

## Day 12: Convergence Benchmarks & Performance Analysis

**Theme:** Comprehensive benchmarking of iterative solvers

**Time estimate:** 8 hours

### Tasks
1. Build convergence benchmark suite:
   - For each SuiteSparse matrix, record:
     - CG iterations (unpreconditioned, ILU-preconditioned, Cholesky-preconditioned) — SPD matrices only
     - GMRES iterations (unpreconditioned, ILU-preconditioned) — all matrices
     - Direct solve time (LU or Cholesky)
     - Iterative solve time (setup + iterations)
     - Final residual norm
2. Generate convergence plots data:
   - Record residual norm vs iteration for CG and GMRES on select matrices
   - Show how preconditioning accelerates convergence
3. Update `bench_main.c`:
   - Add `--iterative` flag for iterative solver benchmarks
   - Report: solver, matrix, preconditioner, iterations, time, residual
4. Write `planning/SPRINT_5/benchmark_results.md`:
   - Iterative vs direct comparison table
   - Preconditioning effectiveness data
   - Parallel SpMV scaling data
   - Recommendations for when to use iterative vs direct

### Deliverables
- Convergence benchmark data for all solvers × all matrices
- Updated benchmark tool with iterative solver support
- Benchmark results document with analysis

### Completion Criteria
- Benchmarks run to completion
- ILU preconditioning reduces iteration count on all test matrices
- Analysis document provides clear solver selection guidance

---

## Day 13: Documentation & Hardening

**Theme:** Update all documentation, harden edge cases

**Time estimate:** 8 hours

### Tasks
1. Update `docs/algorithm.md`:
   - Add CG algorithm section (algorithm, convergence theory, SPD requirement)
   - Add GMRES section (Arnoldi process, Givens rotations, restart strategy)
   - Add ILU(0) section (algorithm, sparsity pattern preservation, limitations)
   - Add parallel SpMV section (row partitioning, scheduling, expected speedup)
   - Add preconditioning section (why, how, preconditioner selection guide)
2. Update `README.md`:
   - Add CG, GMRES, ILU, parallel SpMV to feature list
   - Update API overview table (add `sparse_iterative.h`, `sparse_ilu.h`)
   - Update project structure
   - Update test counts
   - Add iterative solver usage example
3. Edge-case hardening:
   - Very ill-conditioned systems → verify CG/GMRES report non-convergence gracefully
   - Singular systems → verify GMRES handles breakdown
   - Very large restart value → verify no excessive memory allocation
   - ILU on singular matrix → verify proper error handling
4. Run `make test`, `make sanitize`, `make bench` — all clean

### Deliverables
- Updated algorithm documentation with iterative solver coverage
- Updated README
- Edge-case hardening tests
- Clean sanitizer and benchmark runs

### Completion Criteria
- Documentation covers all new features with algorithm descriptions
- All edge cases handled with proper error codes
- `make test`, `make sanitize`, `make bench` all clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final validation, cleanup, and retrospective

**Time estimate:** 6 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run, no crashes
   - OpenMP build: `make omp && make test` — all pass
2. Code review pass:
   - All new public API functions have doc comments
   - All new error codes handled in `sparse_strerror()`
   - `const` correctness on all new functions
   - No compiler warnings with strict flags
3. Verify backward compatibility:
   - Existing code using LU/Cholesky factor/solve works unchanged
   - No breaking API changes
4. Write `planning/SPRINT_5/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, assertion count, API functions, etc.)
   - CG vs GMRES vs direct comparison data
   - Preconditioning effectiveness summary
   - Parallel SpMV scaling results
   - Items deferred to Sprint 6

### Deliverables
- All tests pass under all sanitizers
- Updated README with new API surface
- Sprint retrospective document
- Clean git history with meaningful commits

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 UBSan findings
- `make bench` completes without error
- OpenMP build and tests clean
- README reflects current API
- Retrospective written with honest assessment

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | CG — API design & infrastructure | 8 | `sparse_iterative.h`, types, stubs, test helpers |
| 2 | CG — core algorithm | 10 | `sparse_solve_cg()`, basic tests |
| 3 | CG — validation & convergence | 9 | SuiteSparse validation, convergence tests, CG vs Cholesky |
| 4 | GMRES — Arnoldi & core algorithm | 10 | Arnoldi process, Givens rotations, basic GMRES |
| 5 | GMRES — restart & preconditioning | 10 | Restarted GMRES(k), left preconditioning, ≥5 tests |
| 6 | GMRES — SuiteSparse validation | 8 | Real-matrix validation, restart comparison, GMRES vs LU |
| 7 | ILU(0) — implementation | 10 | `sparse_ilu_factor()`, `sparse_ilu_solve()`, ≥4 tests |
| 8 | ILU(0) — integration & validation | 9 | ILU+CG, ILU+GMRES, Cholesky preconditioner, ≥6 tests |
| 9 | Parallel SpMV — OpenMP | 9 | Parallel SpMV, build system, TSan clean, ≥4 tests |
| 10 | Parallel SpMV — benchmarking | 8 | Thread scaling data, scheduling comparison, integration |
| 11 | Integration testing | 8 | Cross-feature tests, solver comparison, full regression |
| 12 | Convergence benchmarks | 8 | Benchmark results, iterative vs direct analysis |
| 13 | Documentation & hardening | 8 | Algorithm docs, README, edge-case tests |
| 14 | Sprint review & retrospective | 6 | Retrospective, final validation, cleanup |

**Total estimate:** 121 hours (avg ~8.6 hrs/day, max 10 hrs/day)
