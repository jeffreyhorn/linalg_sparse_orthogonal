# Sprint 16 Plan: BiCGSTAB Solver & Iterative Solver Hardening

**Sprint Duration:** 14 days
**Goal:** Add BiCGSTAB for nonsymmetric systems where restarted GMRES is a poor fit, and harden the iterative solver framework with better convergence diagnostics, stagnation detection, and breakdown handling.

**Starting Point:** A sparse linear algebra library with ~20 headers, ~150+ public API functions, ~1145 tests across 37 suites. Iterative solver framework includes CG, GMRES(k), and MINRES — each with single-RHS, block (multi-RHS with per-column convergence), and matrix-free variants (CG and GMRES). Left/right preconditioning via `sparse_precond_fn` callback. ILU(0) and IC(0) preconditioners. Result struct reports iterations, residual norm, and convergence flag. Existing breakdown detection is partial: CG checks `p^T*Ap = 0`, GMRES detects lucky breakdown via `H(j+1,j) < DROP_TOL`, MINRES checks `beta_new <= 0`. No stagnation detection beyond max_iter. No residual history recording. Verbose mode prints to stderr but offers no user callback.

**End State:** `sparse_solve_bicgstab()` solves general nonsymmetric systems with left preconditioning, complementing GMRES for problems where restarts cause convergence stalls. Block and matrix-free BiCGSTAB variants follow the established patterns. All four iterative solvers (CG, GMRES, MINRES, BiCGSTAB) have stagnation detection that reports `SPARSE_ERR_NOT_CONVERGED` early with a stagnation flag when residual fails to decrease for N consecutive iterations. Optional residual history recording stores per-iteration residual norms in a caller-provided array. A verbose callback option replaces stderr printing with user-controlled progress reporting. All solvers are audited for graceful breakdown handling with documented behavior. BiCGSTAB vs GMRES comparison benchmarks on SuiteSparse nonsymmetric matrices.

---

## Day 1: BiCGSTAB — Algorithm Design & Data Structures

**Theme:** Study the BiCGSTAB algorithm and establish internal data structures

**Time estimate:** 10 hours

### Tasks
1. Study the BiCGSTAB algorithm (Van der Vorst, 1992):
   - Two-term recurrence combining BiCG and polynomial stabilization
   - Each iteration requires two SpMVs and two preconditioner applications
   - Avoids the transpose of A (unlike plain BiCG)
   - Smoother convergence than CGS
2. Design BiCGSTAB-specific options:
   - Extend or create `sparse_bicgstab_opts_t` with `max_iter`, `tol`, `verbose`, `precond_fn`, `precond_ctx`
   - Keep consistent with `sparse_iter_opts_t` and `sparse_gmres_opts_t` patterns
3. Design result struct extensions:
   - Add `stagnation` flag field to `sparse_iter_result_t` (or design a new extended result struct)
   - Plan for residual history pointer in options
4. Create `src/sparse_bicgstab_internal.h` with internal workspace struct:
   - Vectors: r, r_hat, p, v, s, t (6 work vectors of size n)
   - Scalars: rho, rho_old, alpha, omega
5. Write scaffolding in `src/sparse_iterative.c` (or new `src/sparse_bicgstab.c`):
   - Function skeleton for `sparse_solve_bicgstab()`
   - Workspace allocation/deallocation helpers
6. Add to Makefile and CMakeLists.txt
7. Run `make format && make lint && make test` — all clean

### Deliverables
- BiCGSTAB algorithm design document (in code comments)
- Internal workspace struct and helpers
- Function skeleton with parameter validation
- Build system integration

### Completion Criteria
- Skeleton compiles and links
- All existing tests still pass
- `make format && make lint && make test` clean

---

## Day 2: BiCGSTAB — Core Solver Implementation

**Theme:** Implement the BiCGSTAB iteration loop

**Time estimate:** 12 hours

### Tasks
1. Implement the BiCGSTAB main loop:
   - Choose arbitrary r_hat_0 (set r_hat = r_0, the initial residual)
   - For each iteration:
     - rho = r_hat^T * r (check rho != 0, breakdown otherwise)
     - beta = (rho / rho_old) * (alpha / omega)
     - p = r + beta * (p - omega * v)
     - Apply preconditioner: p_hat = M^{-1} * p
     - v = A * p_hat
     - alpha = rho / (r_hat^T * v)
     - s = r - alpha * v
     - Check if ||s|| is small enough (early termination)
     - Apply preconditioner: s_hat = M^{-1} * s
     - t = A * s_hat
     - omega = (t^T * s) / (t^T * t) (check omega != 0)
     - x = x + alpha * p_hat + omega * s_hat
     - r = s - omega * t
     - Check convergence: ||r|| / ||b|| < tol
2. Implement left preconditioning via `sparse_precond_fn` callback:
   - Apply M^{-1} to p and s vectors
   - Skip if no preconditioner provided
3. Compute true residual norm for convergence check (||b - Ax|| / ||b||)
4. Write initial tests:
   - 3×3 and 5×5 nonsymmetric systems with known solutions
   - Identity matrix (should converge in 1 iteration)
   - Diagonal matrix
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Complete BiCGSTAB iteration loop
- Left preconditioning support
- True residual convergence verification
- Initial correctness tests

### Completion Criteria
- BiCGSTAB converges on small test systems
- Solutions match direct solve to tolerance
- `make format && make lint && make test` clean

---

## Day 3: BiCGSTAB — Public API & Integration

**Theme:** Expose BiCGSTAB as a public API and test on real matrices

**Time estimate:** 10 hours

### Tasks
1. Add `sparse_solve_bicgstab()` declaration to `include/sparse_iterative.h`:
   - `sparse_err_t sparse_solve_bicgstab(const SparseMatrix *A, const double *b, double *x, const sparse_iter_opts_t *opts, sparse_iter_result_t *result)`
   - Doxygen documentation: algorithm description, when to use vs GMRES, preconditioner support
2. Add `sparse_solve_bicgstab()` to the public API table in documentation
3. Test on SuiteSparse nonsymmetric matrices:
   - west0067 (67×67, chemical engineering)
   - steam1 (if nonsymmetric)
   - Compare convergence rate with GMRES(30) on same system
4. Test with ILU(0) preconditioning:
   - Factor with `sparse_ilu_factor()`
   - Use `sparse_ilu_precond` callback
   - Verify faster convergence than unpreconditioned
5. Edge case tests:
   - Zero RHS (x should be zero)
   - Already-converged initial guess
   - 1×1 system
   - max_iter = 0
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_solve_bicgstab()` public API
- Doxygen documentation
- SuiteSparse integration tests
- Preconditioning tests

### Completion Criteria
- BiCGSTAB produces correct solutions on SuiteSparse matrices
- ILU(0) preconditioning accelerates convergence
- All edge cases handled without crashes
- `make format && make lint && make test` clean

---

## Day 4: BiCGSTAB — Numerical Hardening

**Theme:** Harden BiCGSTAB against numerical edge cases

**Time estimate:** 8 hours

### Tasks
1. Add NaN/Inf detection after each iteration:
   - Check rho, alpha, omega for NaN/Inf
   - Return `SPARSE_ERR_NUMERIC` with diagnostic info
2. Handle near-zero omega:
   - When omega approaches zero, the stabilization polynomial fails
   - Fall back to half-step (x = x + alpha * p_hat) and restart
   - Document this behavior
3. Test on ill-conditioned systems:
   - High condition number matrices (kappa > 1e10)
   - Nearly singular systems
   - Systems where GMRES converges but BiCGSTAB struggles (and vice versa)
4. Add parameter validation:
   - NULL checks for required pointers
   - n > 0 validation
   - Tolerance > 0 validation
5. Verify convergence behavior with different initial guesses:
   - Zero initial guess (default)
   - Random initial guess
   - Near-solution initial guess
6. Run `make format && make lint && make test` — all clean

### Deliverables
- NaN/Inf detection and reporting
- Near-zero omega handling
- Ill-conditioned system tests
- Complete parameter validation

### Completion Criteria
- No crashes or hangs on ill-conditioned inputs
- Clear error codes for numerical failures
- `make format && make lint && make test` clean

---

## Day 5: Block BiCGSTAB — Design & Implementation

**Theme:** Implement multi-RHS block BiCGSTAB with per-column convergence

**Time estimate:** 10 hours

### Tasks
1. Study the block BiCGSTAB pattern established by `sparse_cg_solve_block()` and `sparse_gmres_solve_block()`:
   - Per-column convergence tracking
   - Shared SpMV across columns
   - Each column updates independently once converged
2. Implement `sparse_bicgstab_solve_block()`:
   - Signature: `sparse_err_t sparse_bicgstab_solve_block(const SparseMatrix *A, idx_t nrhs, const double *B, idx_t ldb, double *X, idx_t ldx, const sparse_iter_opts_t *opts, sparse_iter_result_t *result)`
   - Column-major layout for B and X (matching existing block solvers)
   - Per-column convergence: skip converged columns in subsequent iterations
   - Report aggregate result (max iterations, worst residual)
3. Allocate per-column workspace:
   - r, r_hat, p, v, s, t arrays for each active RHS column
4. Write tests:
   - 2-RHS and 4-RHS nonsymmetric systems
   - Verify each column converges independently
   - One column converges fast, another slowly — verify correct handling
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_bicgstab_solve_block()` implementation
- Per-column convergence tracking
- Multi-RHS tests

### Completion Criteria
- Block BiCGSTAB produces correct solutions for all RHS columns
- Per-column convergence works (early columns don't block late ones)
- Results match single-RHS BiCGSTAB applied to each column
- `make format && make lint && make test` clean

---

## Day 6: Block BiCGSTAB Completion & Matrix-Free BiCGSTAB

**Theme:** Finish block variant edge cases and implement matrix-free BiCGSTAB

**Time estimate:** 8 hours

### Tasks
1. Harden block BiCGSTAB:
   - Test with nrhs = 1 (should match single-RHS variant)
   - Test with nrhs = 0 (no-op)
   - ILU(0) preconditioning with block variant
   - Error propagation: callback error stops all columns
2. Implement `sparse_solve_bicgstab_mf()`:
   - Signature: `sparse_err_t sparse_solve_bicgstab_mf(sparse_matvec_fn matvec, const void *ctx, idx_t n, const double *b, double *x, const sparse_iter_opts_t *opts, sparse_iter_result_t *result)`
   - Follow the pattern of `sparse_solve_cg_mf()` and `sparse_solve_gmres_mf()`
   - Replace SpMV with user callback
   - Full preconditioner support
3. Write matrix-free tests:
   - Wrap a SparseMatrix in a matvec callback, verify results match matrix-based BiCGSTAB
   - Custom operator test (e.g., scaled identity)
   - Error propagation from matvec callback
4. Add declarations to `include/sparse_iterative.h` with Doxygen docs
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Hardened block BiCGSTAB
- `sparse_solve_bicgstab_mf()` matrix-free variant
- Consistency tests between all three BiCGSTAB variants

### Completion Criteria
- Matrix-free BiCGSTAB matches matrix-based on identical problems
- Block edge cases handled
- `make format && make lint && make test` clean

---

## Day 7: Stagnation Detection — Framework & CG/MINRES

**Theme:** Design the stagnation detection framework and implement for CG and MINRES

**Time estimate:** 10 hours

### Tasks
1. Design the stagnation detection mechanism:
   - Track residual norms over a sliding window of N iterations (default N = 10)
   - Stagnation condition: max(residuals[window]) / min(residuals[window]) < stagnation_factor (e.g., 1.01)
   - Alternative: residual hasn't decreased by more than stagnation_tol in N iterations
   - Add `stagnation_window` and `stagnation_tol` fields to `sparse_iter_opts_t`
2. Extend `sparse_iter_result_t`:
   - Add `int stagnated` flag (1 if stagnation detected, 0 otherwise)
   - When stagnation detected, return `SPARSE_ERR_NOT_CONVERGED` early with stagnated = 1
3. Implement stagnation detection in CG:
   - Insert check after convergence test each iteration
   - Ring buffer of last N residual norms
   - Early exit with stagnation flag
4. Implement stagnation detection in MINRES:
   - Same ring buffer approach
   - Account for MINRES's guaranteed monotonic residual decrease
5. Write tests:
   - Construct a system that causes CG to stagnate (near-singular, bad conditioning)
   - Verify stagnation is detected within expected iteration range
   - Verify stagnation flag is set in result
   - Verify non-stagnating systems do NOT trigger false positives
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Stagnation detection framework (ring buffer, config options)
- `stagnated` flag in result struct
- CG stagnation detection
- MINRES stagnation detection
- Stagnation tests for both solvers

### Completion Criteria
- Stagnation correctly detected on designed-to-stagnate systems
- No false positives on well-behaved systems
- Early exit saves iterations vs running to max_iter
- `make format && make lint && make test` clean

---

## Day 8: Stagnation Detection — GMRES & BiCGSTAB

**Theme:** Add stagnation detection to GMRES and BiCGSTAB

**Time estimate:** 10 hours

### Tasks
1. Implement stagnation detection in GMRES:
   - Track residual across restarts (not within a single restart cycle)
   - Stagnation often manifests as restarts not improving the residual
   - Use the true residual computed after each restart cycle
2. Implement stagnation detection in BiCGSTAB:
   - BiCGSTAB can exhibit erratic convergence — use the ring buffer approach
   - Track true residual norms (not the recurrence residual)
3. Implement stagnation detection in block variants:
   - Per-column stagnation tracking
   - A column that stagnates is marked done with stagnated flag
   - Aggregate stagnated flag: set if any column stagnated
4. Write tests:
   - GMRES: system where restarts cause stagnation (e.g., highly nonsymmetric with small restart)
   - BiCGSTAB: system with erratic convergence
   - Block solver: one column stagnates, others converge
5. Test backward compatibility:
   - Default stagnation_window = 0 means disabled (existing behavior unchanged)
   - All existing tests pass with stagnation detection defaulting to off
6. Run `make format && make lint && make test` — all clean

### Deliverables
- GMRES stagnation detection (across restarts)
- BiCGSTAB stagnation detection
- Block solver per-column stagnation
- Backward-compatible defaults

### Completion Criteria
- All four solvers have stagnation detection
- Default behavior unchanged (backward compatible)
- Block solvers handle per-column stagnation
- `make format && make lint && make test` clean

---

## Day 9: Convergence Diagnostics — Residual History Recording

**Theme:** Add optional per-iteration residual history recording to all iterative solvers

**Time estimate:** 10 hours

### Tasks
1. Design the residual history API:
   - Add `double *residual_history` and `idx_t residual_history_len` fields to options structs
   - Caller allocates the array (sized to max_iter)
   - Solver fills in residual_history[i] = ||r_i|| / ||b|| for each iteration i
   - Actual number of entries stored returned in result struct
2. Implement residual history recording in CG:
   - Store relative residual norm after each iteration
   - If array is shorter than iterations, stop recording (don't overflow)
3. Implement residual history recording in GMRES:
   - Store the QR estimate within each restart, true residual at restart boundaries
   - Clearly document which residual is stored at each index
4. Implement residual history recording in MINRES:
   - Store the monotonically decreasing residual norm each iteration
5. Implement residual history recording in BiCGSTAB:
   - Store the true residual after each full iteration (both half-steps)
6. Write tests:
   - Record history, verify it matches manually computed residuals
   - Verify monotonic decrease for CG (SPD) and MINRES
   - Verify history length matches iteration count
   - NULL history pointer means no recording (backward compatible)
7. Run `make format && make lint && make test` — all clean

### Deliverables
- Residual history recording for all four solvers
- `residual_history` / `residual_history_len` in options
- `residual_history_count` in result struct
- History correctness tests

### Completion Criteria
- Residual history correctly recorded for all solvers
- History matches independently verified residual norms
- No recording when pointer is NULL (backward compatible)
- `make format && make lint && make test` clean

---

## Day 10: Convergence Diagnostics — Verbose Callback

**Theme:** Add a user-supplied verbose callback for custom progress reporting

**Time estimate:** 8 hours

### Tasks
1. Design the verbose callback interface:
   - `typedef void (*sparse_iter_callback_fn)(const sparse_iter_progress_t *progress, void *ctx)`
   - `sparse_iter_progress_t` struct: iteration, residual_norm, solver_name, solver-specific data
   - Add `callback_fn` and `callback_ctx` to options structs
2. Replace existing stderr verbose printing:
   - When callback is provided, call it instead of fprintf(stderr, ...)
   - When callback is NULL and verbose is set, use default stderr printing (backward compatible)
   - When callback is NULL and verbose is off, no output
3. Implement callback invocation in all four solvers:
   - CG: call after each iteration
   - GMRES: call after each inner iteration and each restart
   - MINRES: call after each iteration
   - BiCGSTAB: call after each full iteration
4. Write tests:
   - Custom callback that records calls to an array, verify correct invocations
   - Verify iteration count matches callback call count
   - NULL callback with verbose=1 still prints to stderr
   - NULL callback with verbose=0 produces no output
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_iter_callback_fn` type and progress struct
- Callback support in all four solvers
- Backward-compatible verbose behavior
- Callback invocation tests

### Completion Criteria
- Custom callbacks receive correct per-iteration data
- Existing verbose behavior unchanged when no callback provided
- `make format && make lint && make test` clean

---

## Day 11: Breakdown Handling — CG & GMRES Audit

**Theme:** Audit and improve breakdown handling in CG and GMRES

**Time estimate:** 10 hours

### Tasks
1. Audit CG breakdown conditions:
   - `p^T * A * p = 0`: currently detected — verify behavior is correct
   - `r^T * z = 0` (preconditioned CG): check if handled
   - Near-zero denominators: add threshold-based detection (not just exact zero)
   - Return distinct error code or flag indicating breakdown type
2. Audit GMRES breakdown conditions:
   - Lucky breakdown (`H(j+1,j) ≈ 0`): Krylov subspace contains exact solution
   - Currently detects via `DROP_TOL` — verify early termination extracts correct solution
   - Arnoldi near-zero vector: handle gracefully
   - Restart with zero residual: detect and return success
3. Implement breakdown reporting:
   - Add `int breakdown` flag to result struct (or reuse existing fields)
   - Distinguish between "breakdown = failure" and "lucky breakdown = success"
4. Write tests for each breakdown condition:
   - CG: construct system where p^T*Ap = 0 at some iteration
   - GMRES: construct system where lucky breakdown occurs (solution in Krylov subspace)
   - Near-zero denominator tests with threshold
5. Document breakdown behavior in header comments:
   - When each breakdown type can occur
   - What the solver does in response
   - What the caller should check
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Hardened CG breakdown handling with threshold-based detection
- Verified GMRES lucky breakdown and Arnoldi breakdown handling
- Breakdown reporting in result struct
- Breakdown-triggering test cases
- Documented breakdown behavior

### Completion Criteria
- All identified breakdown conditions handled gracefully
- Lucky breakdown correctly returns the exact solution
- No crashes on any breakdown scenario
- `make format && make lint && make test` clean

---

## Day 12: Breakdown Handling — MINRES & BiCGSTAB

**Theme:** Audit and improve breakdown handling in MINRES and BiCGSTAB

**Time estimate:** 10 hours

### Tasks
1. Audit MINRES breakdown conditions:
   - Lanczos breakdown: `beta_{k+1} = 0` means invariant subspace found
   - Currently checks `beta_new <= 0` — verify this correctly handles exact zero
   - Near-zero beta: add threshold-based detection
   - Verify solution is extracted correctly on Lanczos breakdown
2. Audit BiCGSTAB breakdown conditions:
   - `rho = 0`: `r_hat^T * r = 0`, BiCG breakdown — solver cannot continue
   - `omega = 0`: stabilization step failed — half-step may still be useful
   - Implement graceful handling:
     - rho = 0: return current best solution with breakdown flag
     - omega = 0: accept the half-step (x += alpha * p_hat), set breakdown flag
3. Write tests for each breakdown:
   - MINRES: system where Lanczos terminates early (solution in Krylov subspace)
   - BiCGSTAB rho = 0: construct system that triggers this
   - BiCGSTAB omega = 0: construct system where stabilization fails
4. Document all breakdown behaviors:
   - Create a breakdown behavior table in `sparse_iterative.h` docs
   - For each solver: list conditions, detection method, response, caller action
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Hardened MINRES Lanczos breakdown handling
- BiCGSTAB rho = 0 and omega = 0 handling
- Breakdown test cases for both solvers
- Comprehensive breakdown documentation table

### Completion Criteria
- MINRES and BiCGSTAB handle all breakdown conditions gracefully
- Breakdown flag correctly set in result struct
- Documentation covers all four solvers' breakdown behavior
- `make format && make lint && make test` clean

---

## Day 13: Integration Tests, Benchmarks & Documentation

**Theme:** BiCGSTAB vs GMRES comparison, comprehensive testing, and documentation updates

**Time estimate:** 8 hours

### Tasks
1. BiCGSTAB vs GMRES comparison benchmarks:
   - Add `benchmarks/bench_bicgstab.c`
   - Compare on SuiteSparse nonsymmetric matrices: iteration count, time, final residual
   - Test with and without ILU(0) preconditioning
   - Report which solver wins on each matrix and by how much
2. Test stagnation detection across all solvers:
   - Unified test with multiple solvers on the same stagnation-inducing system
   - Verify stagnation window parameter works correctly
3. Test convergence diagnostics end-to-end:
   - Record residual history, feed to a custom verbose callback, verify consistency
4. Update README iterative solver section:
   - Add BiCGSTAB to feature list and solver table
   - Add stagnation detection, residual history, verbose callback to feature list
   - Update API function count
   - Add guidance: when to use CG vs GMRES vs MINRES vs BiCGSTAB
5. Update `docs/algorithm.md`:
   - Add BiCGSTAB algorithm description
   - Add stagnation detection description
6. Run `make format && make lint && make test` — all clean

### Deliverables
- BiCGSTAB vs GMRES comparison benchmark
- Cross-solver stagnation and diagnostics tests
- Updated README and algorithm documentation
- Solver selection guidance

### Completion Criteria
- Benchmarks demonstrate BiCGSTAB's strengths and weaknesses vs GMRES
- All new features tested end-to-end
- Documentation is accurate and complete
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final testing, metrics collection, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Final metrics collection:
   - Total test count (target: ~1250+)
   - BiCGSTAB-specific test count
   - Stagnation detection test count
   - Convergence diagnostics test count
   - Breakdown handling test count
2. Full regression:
   - `make clean && make format && make lint && make test`
   - `make sanitize` (UBSan)
   - `make examples` and run each
   - `make bench`
3. Verify all Sprint 16 deliverables:
   - `sparse_solve_bicgstab()`, block, and matrix-free variants
   - Stagnation detection in all iterative solvers
   - Optional residual history recording
   - Verbose callback option
   - Breakdown handling audited and documented
   - BiCGSTAB vs GMRES comparison benchmarks
4. Write `docs/planning/EPIC_2/SPRINT_16/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Final metrics
   - Items deferred (if any)
5. Update project plan if any Sprint 16 items were deferred

### Deliverables
- Sprint retrospective document
- Updated metrics
- Clean final build

### Completion Criteria
- All Sprint 16 items complete or explicitly deferred
- Retrospective written with honest assessment
- `make format && make lint && make test` clean
