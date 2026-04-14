# Sprint 15 Plan: COLAMD Ordering & QR Minimum-Norm Least Squares

**Sprint Duration:** 14 days
**Goal:** Upgrade the ordering stack with COLAMD for unsymmetric/QR problems and add minimum-norm least-squares solves for underdetermined systems.

**Starting Point:** A sparse linear algebra library with 17 headers, ~140 public API functions, 1069+ tests across 36 suites, symbolic analysis / numeric factorization split (Sprint 14), AMD/RCM fill-reducing reordering, column-pivoted QR with Householder reflections, least-squares solve, rank estimation, null-space extraction, and economy QR. The current QR column ordering uses AMD on A^T*A as a proxy, which is suboptimal for unsymmetric column structure. QR solve handles overdetermined systems (m >= n) but not minimum-norm for underdetermined (m < n).

**End State:** `sparse_reorder_colamd()` computes column approximate minimum degree ordering directly from the column adjacency graph without forming A^T*A. COLAMD is integrated as the default QR column ordering, with benchmarks showing fill reduction vs AMD proxy. `sparse_qr_solve_minnorm()` computes minimum 2-norm solutions for underdetermined systems via QR of A^T. Rank-revealing diagnostics are improved: R diagonal exposure, rank-deficiency warnings, threshold guidance. Full test coverage on SuiteSparse unsymmetric matrices and known underdetermined systems.

---

## Day 1: COLAMD — Algorithm Design & Column Adjacency Graph

**Theme:** Implement the column adjacency graph construction and COLAMD data structures

**Time estimate:** 10 hours

### Tasks
1. Study the COLAMD algorithm (Davis et al., "A column approximate minimum degree ordering algorithm"):
   - Operates on the column adjacency graph: columns i and j are adjacent if they share a nonzero row
   - Uses aggressive absorption and mass elimination to approximate minimum degree
   - Works directly on A's row structure without forming A^T*A
2. Design internal data structures in `src/sparse_colamd_internal.h`:
   - Row and column score arrays
   - Degree lists (hash-based approximate degree)
   - Supernode/mass elimination tracking
3. Implement column adjacency graph construction from A's row structure:
   - Build row-index lists per column (transpose of CSR-like view)
   - This replaces the need to explicitly form A^T*A
4. Write scaffolding tests:
   - Column adjacency for small known matrices
   - Verify adjacency is symmetric
5. Add to Makefile and CMakeLists.txt
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `src/sparse_colamd_internal.h` with COLAMD data structures
- Column adjacency graph construction
- Scaffolding tests

### Completion Criteria
- Column adjacency graph is correct for small test matrices
- `make format && make lint && make test` clean

---

## Day 2: COLAMD — Core Elimination Algorithm

**Theme:** Implement the COLAMD column ordering algorithm

**Time estimate:** 12 hours

### Tasks
1. Implement the COLAMD ordering loop:
   - Initialize column scores (approximate degrees)
   - Build degree lists (bucket sort by score)
   - Main loop: select minimum-degree column, perform mass elimination
   - Aggressive absorption: merge indistinguishable columns
   - Update scores for affected columns
2. Handle edge cases:
   - Empty columns (zero degree)
   - Dense rows (rows with > dense_threshold nonzeros — skip these)
   - Rectangular matrices (m != n)
3. Implement `sparse_reorder_colamd_internal(const SparseMatrix *A, idx_t *perm)`:
   - Internal function returning column permutation
   - perm[new] = old convention (matching existing reorder routines)
4. Write tests:
   - Small matrices with known optimal orderings
   - Verify perm is a valid permutation
   - Compare with brute-force degree computation

### Deliverables
- Complete COLAMD elimination algorithm
- Internal ordering function
- Tests for small known matrices

### Completion Criteria
- COLAMD produces valid permutations for all test matrices
- Degree computation matches brute-force verification
- `make format && make lint && make test` clean

---

## Day 3: COLAMD — Dense Row Detection & Tuning

**Theme:** Handle dense rows, tune thresholds, and harden the implementation

**Time estimate:** 10 hours

### Tasks
1. Implement dense row detection:
   - Rows with nnz > sqrt(n) * DENSE_SCALE are treated as dense
   - Dense rows are excluded from the column adjacency graph
   - They are appended at the end of the ordering
2. Implement configurable knobs:
   - `colamd_dense_row_threshold` — dense row cutoff (default: 10 * sqrt(n))
   - `colamd_aggressive` — enable/disable aggressive absorption (default: on)
3. Add overflow guards and allocation checks (SIZE_MAX validation)
4. Test on pathological inputs:
   - All-dense matrix (every row is dense)
   - Single dense row in otherwise sparse matrix
   - Very tall (m >> n) and very wide (m << n) matrices
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Dense row handling
- Configurable thresholds
- Hardened allocation with overflow checks
- Pathological input tests

### Completion Criteria
- Dense rows are excluded and ordering is still valid
- No crashes on edge cases
- `make format && make lint && make test` clean

---

## Day 4: COLAMD — Public API & Integration

**Theme:** Expose COLAMD as a public reorder API function

**Time estimate:** 10 hours

### Tasks
1. Add `SPARSE_REORDER_COLAMD = 3` to the `sparse_reorder_t` enum in `sparse_types.h`
2. Implement `sparse_reorder_colamd(const SparseMatrix *A, idx_t *perm)` in `sparse_reorder.c`:
   - Public API matching `sparse_reorder_amd`/`sparse_reorder_rcm` signature
   - Delegates to internal COLAMD implementation
   - Handles square and rectangular matrices
3. Add declaration to `include/sparse_reorder.h` with Doxygen docs
4. Write public API tests:
   - Test on unsymmetric matrices from SuiteSparse (west0067, steam1)
   - Verify perm is valid permutation
   - Compare fill-in: COLAMD perm vs no perm vs AMD perm for QR
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_reorder_colamd()` in public API
- `SPARSE_REORDER_COLAMD` enum value
- SuiteSparse integration tests

### Completion Criteria
- COLAMD produces valid orderings on all SuiteSparse test matrices
- `make format && make lint && make test` clean

---

## Day 5: COLAMD — QR Integration

**Theme:** Wire COLAMD into QR factorization as the default column ordering

**Time estimate:** 10 hours

### Tasks
1. Update `sparse_qr_factor_opts()` to support `SPARSE_REORDER_COLAMD`:
   - Add case to the reorder switch
   - Apply column permutation before QR
2. Make COLAMD the recommended column ordering for QR in documentation:
   - Update `sparse_qr.h` header docs
   - Add usage example with COLAMD
3. Write comparison tests:
   - Factor with COLAMD vs AMD vs no reordering
   - Verify all produce correct least-squares solutions
   - Compare R fill-in (nnz(R) with each ordering)
4. Benchmark fill reduction:
   - Add `benchmarks/bench_colamd.c` comparing fill-in on SuiteSparse matrices
   - Report nnz(R) for COLAMD, AMD, and natural ordering
5. Run `make format && make lint && make test` — all clean

### Deliverables
- COLAMD integrated with QR factorization
- Fill-reduction comparison benchmark
- QR solve correctness tests with COLAMD

### Completion Criteria
- QR + COLAMD produces correct solutions
- COLAMD shows fill reduction vs AMD on unsymmetric matrices
- `make format && make lint && make test` clean

---

## Day 6: COLAMD — Wire into Analysis API & Hardening

**Theme:** Integrate COLAMD with sparse_analyze() for LU and harden edge cases

**Time estimate:** 10 hours

### Tasks
1. Update `sparse_analyze()` to support `SPARSE_REORDER_COLAMD`:
   - Add case to the reorder switch for all factorization types
   - For symmetric types (Cholesky/LDL^T), COLAMD computes column ordering on the full matrix, which is valid but suboptimal — document this
2. Update `sparse_factor_numeric()` to handle COLAMD permutations
3. Test analyze+factor with COLAMD on:
   - Unsymmetric LU matrices
   - Symmetric Cholesky matrices (verify still works, compare with AMD)
4. Stress-test COLAMD:
   - 0×0 matrix, 1×1 matrix
   - All-zero matrix
   - Identity matrix
   - Matrix with duplicate entries
5. Run `make format && make lint && make test` — all clean

### Deliverables
- COLAMD integrated with analysis API
- Edge case tests
- LU + COLAMD tests

### Completion Criteria
- `sparse_analyze()` with COLAMD works for all factorization types
- All edge cases handled without crashes
- `make format && make lint && make test` clean

---

## Day 7: QR Minimum-Norm — Algorithm Design

**Theme:** Design and implement the minimum-norm least-squares algorithm for underdetermined systems

**Time estimate:** 12 hours

### Tasks
1. Implement the minimum-norm algorithm for m < n:
   - Compute QR factorization of A^T: A^T * P = Q * R (m×m upper triangular R)
   - The minimum-norm solution is: x = Q * R^{-T} * (P^T * b)
   - This gives the minimum 2-norm solution to the underdetermined system A*x = b
2. Implement `sparse_qr_factor_transpose()` or reuse existing QR on a transposed copy:
   - Build A^T via `sparse_transpose()`
   - Factor A^T with QR (optionally with COLAMD ordering)
3. Implement the solve chain:
   - Permute b: b_p = P^T * b
   - Forward substitute: solve R^T * y = b_p (R^T is lower triangular)
   - Apply Q: x = Q * y
4. Write initial tests:
   - 2×4 underdetermined system with known minimum-norm solution
   - Verify ||x||_2 is minimal among all solutions
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Minimum-norm algorithm implementation
- Forward substitution with R^T
- Initial underdetermined system tests

### Completion Criteria
- Minimum-norm solution is correct for known systems
- ||x||_2 is minimal (compare with x + null-space perturbation)
- `make format && make lint && make test` clean

---

## Day 8: QR Minimum-Norm — Public API

**Theme:** Design and implement the public API for minimum-norm least-squares

**Time estimate:** 10 hours

### Tasks
1. Add `sparse_qr_solve_minnorm()` to `include/sparse_qr.h`:
   - `sparse_err_t sparse_qr_solve_minnorm(const SparseMatrix *A, const double *b, double *x, sparse_qr_opts_t *opts)`
   - Internally: transpose A, factor, solve
   - Return the minimum 2-norm solution
2. Add Doxygen documentation:
   - Explain when to use: underdetermined systems (m < n)
   - Document that for m >= n this is equivalent to regular least-squares
   - Explain the algorithm (QR of A^T)
3. Handle edge cases:
   - m >= n: fall back to regular QR solve
   - Rank-deficient A^T: detect via R diagonal, return warning
   - Empty matrix
4. Write API tests:
   - Various underdetermined sizes (2×4, 3×6, 5×10)
   - Verify A*x = b (or A*x ≈ b for inconsistent systems)
   - Verify ||x||_2 is minimal
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_solve_minnorm()` public API
- Full Doxygen documentation
- Underdetermined system tests

### Completion Criteria
- API produces correct minimum-norm solutions
- Edge cases handled gracefully
- `make format && make lint && make test` clean

---

## Day 9: QR Minimum-Norm — Refinement & Rank Deficiency

**Theme:** Handle rank-deficient underdetermined systems and add iterative refinement

**Time estimate:** 10 hours

### Tasks
1. Handle rank-deficient underdetermined systems:
   - When rank(A) < m, the system may be inconsistent
   - Return the minimum-norm least-squares solution (pseudoinverse behavior)
   - Use rank from QR to determine effective rank
2. Add iterative refinement for minimum-norm:
   - Compute residual r = b - A*x
   - Solve for correction with the same factorization
   - Iterate until convergence
3. Write rank-deficiency tests:
   - System with rank-1 deficiency
   - System with A having a zero row
   - Compare with SVD pseudoinverse solution
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Rank-deficient minimum-norm handling
- Iterative refinement for minimum-norm
- Rank-deficiency tests with SVD comparison

### Completion Criteria
- Rank-deficient systems produce correct minimum-norm solutions
- Iterative refinement improves accuracy
- `make format && make lint && make test` clean

---

## Day 10: Rank-Revealing Improvements — R Diagonal Exposure

**Theme:** Expose the R diagonal and improve rank detection diagnostics

**Time estimate:** 10 hours

### Tasks
1. Add `sparse_qr_diag_r()` to extract the R diagonal:
   - `sparse_err_t sparse_qr_diag_r(const sparse_qr_t *qr, double *diag)`
   - Returns the diagonal of R in factorization order
   - Useful for condition estimation and manual rank determination
2. Add rank-deficiency warning mechanism:
   - `sparse_qr_rank_info()` struct with rank, effective rank, estimated condition
   - Warning when near-rank-deficient (smallest R diagonal close to threshold)
3. Document threshold selection guidance:
   - Add section in `sparse_qr.h` docs explaining how to choose rank tolerance
   - Reference machine epsilon, problem scale, and matrix norms
4. Write tests:
   - Extract R diagonal, verify matches direct R access
   - Near-rank-deficient matrix: verify warning triggers
   - Full-rank matrix: verify no warning
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_diag_r()` API
- Rank info struct and diagnostics
- Threshold guidance documentation
- R diagonal and rank warning tests

### Completion Criteria
- R diagonal correctly extracted
- Rank warnings accurate for near-singular matrices
- `make format && make lint && make test` clean

---

## Day 11: Rank-Revealing Improvements — Condition Estimation & Documentation

**Theme:** Add condition number estimation from QR and improve documentation

**Time estimate:** 8 hours

### Tasks
1. Add `sparse_qr_condest()`:
   - Estimate condition number from R diagonal: cond ≈ |R(0,0)| / |R(k-1,k-1)|
   - This is a quick estimate (not full Hager/Higham), but useful for diagnostics
2. Update `sparse_qr_solve()` documentation:
   - Clearly distinguish: overdetermined (least-squares), square (direct), underdetermined (use minnorm)
   - Add `@note` about rank-deficient behavior
   - Add `@see sparse_qr_solve_minnorm` cross-reference
3. Update `sparse_qr_rank()` documentation:
   - Document how tolerance interacts with R diagonal
   - Reference `sparse_qr_diag_r()` for manual inspection
4. Write tests:
   - Condition estimate for well-conditioned and ill-conditioned matrices
   - Verify condest is within an order of magnitude of true condition
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_qr_condest()` quick condition estimator
- Improved QR solve/rank documentation
- Condition estimation tests

### Completion Criteria
- Condition estimate is reasonable for test matrices
- Documentation clearly describes all QR solve modes
- `make format && make lint && make test` clean

---

## Day 12: SuiteSparse Integration Tests & Benchmarks

**Theme:** Comprehensive testing on real-world matrices and fill-reduction benchmarks

**Time estimate:** 10 hours

### Tasks
1. Test COLAMD on SuiteSparse unsymmetric matrices:
   - west0067, steam1, and any additional test matrices
   - Verify ordering is valid, measure fill reduction
2. Test minimum-norm on constructed underdetermined systems:
   - Use submatrices of SuiteSparse matrices (take first m rows of an m×n matrix)
   - Verify A*x = b and ||x||_2 minimality
3. Write fill-reduction benchmark:
   - Compare COLAMD vs AMD vs natural for QR R-factor fill
   - Report results in benchmark output
4. Test backward compatibility:
   - All existing QR tests pass with COLAMD available
   - No behavior change when using AMD or no reordering
5. Run `make format && make lint && make test` — all clean

### Deliverables
- SuiteSparse integration tests for COLAMD
- Underdetermined SuiteSparse tests
- Fill-reduction benchmark results
- Backward compatibility verification

### Completion Criteria
- All SuiteSparse tests pass
- Fill reduction demonstrated on real matrices
- `make format && make lint && make test` clean

---

## Day 13: Documentation & Example Programs

**Theme:** Document the new APIs and create example programs

**Time estimate:** 8 hours

### Tasks
1. Update README:
   - Add COLAMD to feature list and reordering section
   - Add minimum-norm to QR feature list
   - Add rank diagnostics to feature list
   - Update API table with new functions
   - Update key functions list
2. Update `docs/algorithm.md`:
   - Add COLAMD algorithm description
   - Add minimum-norm algorithm description
   - Document when to use COLAMD vs AMD vs RCM
3. Create `examples/example_minnorm.c`:
   - Demonstrate underdetermined system solve
   - Show comparison with regular QR solve
   - Show rank diagnostics
4. Create `examples/example_colamd.c`:
   - Demonstrate COLAMD ordering for QR
   - Show fill-reduction comparison
5. Add examples to CMakeLists.txt
6. Run `make format && make lint && make test` — all clean

### Deliverables
- README and algorithm docs updated
- Working example programs
- Build system updated

### Completion Criteria
- Examples compile, run, and demonstrate features
- Documentation accurately describes new APIs
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final testing, metrics collection, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Final metrics collection:
   - Total test count
   - COLAMD-specific test count
   - Minimum-norm test count
   - Fill-reduction benchmark data
2. Full regression:
   - `make clean && make format && make lint && make test`
   - `make sanitize` (UBSan)
   - `make examples` and run each
   - `make bench`
3. Write `docs/planning/EPIC_2/SPRINT_15/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Final metrics
   - Items deferred (if any)
4. Update project plan if any Sprint 15 items were deferred

### Deliverables
- Sprint retrospective document
- Updated metrics
- Clean final build

### Completion Criteria
- All Sprint 15 items complete or explicitly deferred
- Retrospective written with honest assessment
- `make format && make lint && make test` clean
