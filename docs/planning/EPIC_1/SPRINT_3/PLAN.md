# Sprint 3 Plan: Numerical Robustness & Fill-Reducing Reordering

**Sprint Duration:** 14 days
**Goal:** Add condition number estimation for diagnostics and implement fill-reducing reordering (AMD/RCM) to improve factorization performance on larger matrices.

**Starting Point:** A sparse LU library with 30 public API functions, 192 tests across 9 test suites, 6 SuiteSparse reference matrices, relative drop tolerance, and arithmetic operations. steam1 and orsirr_1 show 10-15x fill-in without reordering, motivating this sprint's reordering work.

**End State:** `sparse_lu_condest()` provides 1-norm condition estimates from existing LU factors with ill-conditioning warnings. Fill-reducing reordering via AMD and/or RCM is integrated as an optional pre-processing step, with benchmark data demonstrating fill-in reduction on real-world matrices.

---

## Day 1: Condition Number Estimation — Research & API Design

**Theme:** Design the condition estimator interface and study Hager's algorithm

**Time estimate:** 5 hours

### Tasks
1. Study Hager's 1-norm condition number estimation algorithm (Hager 1984, Higham 2000):
   - The estimator uses the already-computed LU factors to estimate `||A^{-1}||_1` without forming the inverse
   - Requires solving `A*x = e` and `A^T*y = z` for selected right-hand sides
   - Typically converges in 2-5 iterations
2. Design the public API:
   - `sparse_err_t sparse_lu_condest(const SparseMatrix *LU, double *condest)` — estimate 1-norm condition number from LU factors
   - Requires the matrix to have been factored (check for `row_perm`/`col_perm` presence)
   - Returns `SPARSE_ERR_BADARG` if matrix has not been factored
3. Design the internal requirement: need `sparse_lu_solve_transpose()` or equivalent to solve `A^T * y = z`
   - This means implementing forward/backward substitution in transposed order: `U^T * L^T * y = P^T * z`
   - Evaluate whether to expose this as a public API or keep it internal
4. Document the design decisions in code comments at the declaration site
5. Add `sparse_lu_condest()` declaration to `include/sparse_lu.h` (stub implementation returning `SPARSE_ERR_BADARG`)

### Deliverables
- API declaration for `sparse_lu_condest()` in header
- Stub implementation that compiles cleanly
- Design notes in code comments covering algorithm choice and API decisions

### Completion Criteria
- Code compiles with zero warnings
- Existing `make test` passes (stub does not break anything)

---

## Day 2: Condition Number Estimation — Transpose Solve

**Theme:** Implement transpose solve needed for Hager's algorithm

**Time estimate:** 6 hours

### Tasks
1. Implement `sparse_lu_solve_transpose()` (internal or public) in `src/sparse_lu.c`:
   - Given LU factors with permutations P and Q, solve `A^T * x = b`
   - This is equivalent to solving `(PLU Q)^T x = b`, i.e., `Q^T U^T L^T P^T x = b`
   - Steps: apply Q permutation to b, forward-substitute with U^T, backward-substitute with L^T, apply P^{-1} permutation
2. Handle the permuted LU storage format:
   - L is stored below the diagonal in the factored matrix (unit diagonal)
   - U is stored on/above the diagonal
   - `row_perm` and `col_perm` encode P and Q
3. Write unit tests for transpose solve:
   - Small 3x3 matrix: factor, solve A*x=b and A^T*y=c, verify both give correct results
   - Identity matrix: transpose solve should equal forward solve
   - Known matrix from test corpus: verify `A^T * x_T = b` produces correct residual
4. Run `make test` — all pass

### Deliverables
- Working transpose solve implementation
- ≥3 tests validating transpose solve correctness
- Clean compilation with no warnings

### Completion Criteria
- Transpose solve produces correct results on test cases
- `A^T * (transpose_solve(b))` residual < 1e-12 for well-conditioned test matrices
- All existing tests still pass

---

## Day 3: Condition Number Estimation — Hager's Algorithm

**Theme:** Implement the 1-norm condition estimator

**Time estimate:** 6 hours

### Tasks
1. Implement Hager's algorithm in `sparse_lu_condest()`:
   - Compute `||A||_1` (1-norm = max column sum of |a_ij|) — add `sparse_norm1()` or compute inline
   - Estimate `||A^{-1}||_1` using the iterative Hager/Higham algorithm:
     - Start with x = [1/n, 1/n, ..., 1/n]
     - Iterate: solve A*w = x, compute xi = sign(w), solve A^T*z = xi
     - If ||z||_inf <= z^T * w, converged: estimate = ||w||_1
     - Otherwise: x = e_j where j = argmax|z_j|, repeat
   - Limit iterations (e.g., max 5) to bound cost
   - condest = ||A||_1 * ||A^{-1}||_1_estimate
2. Handle edge cases:
   - Zero or near-zero pivots encountered during estimation solves
   - Already-singular matrix (should have been caught at factor time)
   - 1x1 matrix (trivial condition number)
3. Write unit tests:
   - Identity matrix → condest = 1.0
   - Diagonal matrix with entries [1, 2, 3] → condest = 3.0 (exact)
   - Known ill-conditioned matrix (e.g., Hilbert-like) → condest >> 1
   - Well-conditioned tridiagonal → condest modest (< 100)
4. Run `make test` — all pass

### Deliverables
- Working `sparse_lu_condest()` implementation
- ≥4 unit tests for condition estimation
- Correct results on known-condition matrices

### Completion Criteria
- condest(I) = 1.0
- condest on ill-conditioned matrix is large (> 1e6)
- condest on well-conditioned matrix is small (< 100)
- `make test` and `make sanitize` clean

---

## Day 4: Condition Number Estimation — Warning System & Polish

**Theme:** Add ill-conditioning warnings and validate on SuiteSparse matrices

**Time estimate:** 5 hours

### Tasks
1. Add optional warning callback or threshold-based warning:
   - Add `sparse_lu_condest_warn()` or a simpler approach: `sparse_lu_factor()` optionally computes condest and prints a warning to stderr when condest exceeds a threshold (e.g., 1e12)
   - Consider a simple approach: add a `double condest` output field accessible after factorization, let the caller decide what to do with it
2. Run `sparse_lu_condest()` on all 6 SuiteSparse matrices:
   - Record condition estimates
   - Compare with residual norms from Sprint 2 — do high condest values correlate with larger residuals?
3. Add condest to benchmark output in `bench_main.c`:
   - Report condition estimate alongside timing and residual data
4. Update `include/sparse_lu.h` doc comments with usage examples
5. Run full test suite and sanitizers

### Deliverables
- Condition estimate accessible after factorization
- Condest data for all SuiteSparse matrices
- Benchmark output includes condition estimate
- Updated API documentation

### Completion Criteria
- Condest results correlate with observed residual norms
- `make test` and `make sanitize` clean
- API documentation complete

---

## Day 5: Fill-Reducing Reordering — Research & Data Structures

**Theme:** Design reordering infrastructure and study AMD/RCM algorithms

**Time estimate:** 5 hours

### Tasks
1. Study the Reverse Cuthill-McKee (RCM) algorithm:
   - BFS-based bandwidth reduction
   - Simpler to implement than AMD
   - Good for banded/structured matrices
   - Input: adjacency graph of A+A^T (symmetrized sparsity pattern)
2. Study the Approximate Minimum Degree (AMD) algorithm:
   - Greedy elimination ordering that minimizes fill-in
   - More complex but generally better fill-in reduction
   - Key data structures: quotient graph, degree lists, element/variable distinction
3. Design the reordering API:
   - `sparse_err_t sparse_reorder_rcm(const SparseMatrix *A, idx_t *perm)` — compute RCM permutation
   - `sparse_err_t sparse_reorder_amd(const SparseMatrix *A, idx_t *perm)` — compute AMD permutation
   - `sparse_err_t sparse_permute(const SparseMatrix *A, const idx_t *row_perm, const idx_t *col_perm, SparseMatrix **B)` — apply permutation to create reordered matrix
4. Design the factorization integration:
   - Add `sparse_reorder_t` enum: `SPARSE_REORDER_NONE`, `SPARSE_REORDER_RCM`, `SPARSE_REORDER_AMD`
   - Plan how reordering integrates with `sparse_lu_factor()` — either as a pre-processing step or via an options struct
5. Add declarations to headers (stub implementations)

### Deliverables
- Reordering API declarations in headers
- `sparse_reorder_t` enum in `sparse_types.h`
- `sparse_permute()` declaration
- Stub implementations that compile cleanly
- Design notes documenting algorithm choices

### Completion Criteria
- Code compiles with zero warnings
- Existing `make test` passes
- API design documented in header comments

---

## Day 6: Fill-Reducing Reordering — Graph Extraction & Permute

**Theme:** Build adjacency graph from sparse matrix and implement permutation

**Time estimate:** 6 hours

### Tasks
1. Implement `sparse_permute()` in `src/sparse_matrix.c`:
   - Given row_perm and col_perm arrays, create a new matrix B where B(i,j) = A(row_perm[i], col_perm[j])
   - Handle symmetric permutation (row_perm == col_perm) as a special case
   - Validate permutation arrays (must be valid permutations of 0..n-1)
2. Implement internal helper to extract the adjacency graph of A+A^T:
   - For reordering, we need the symmetrized sparsity pattern
   - Build adjacency lists: for each (i,j) in A with i != j, add edge (i,j) and (j,i)
   - Return as an array of arrays (or CSR-like structure)
3. Write tests for `sparse_permute()`:
   - Identity permutation → output equals input
   - Reverse permutation on diagonal matrix → diagonal reversed
   - Known permutation on small matrix → verify entries are in correct positions
   - NULL inputs → proper error codes
4. Run `make test` — all pass

### Deliverables
- Working `sparse_permute()` implementation with tests
- Internal adjacency graph extraction utility
- ≥4 permutation tests

### Completion Criteria
- `sparse_permute()` produces correct results on all test cases
- Adjacency graph correctly represents symmetrized sparsity pattern
- `make test` and `make sanitize` clean

---

## Day 7: Fill-Reducing Reordering — RCM Implementation

**Theme:** Implement Reverse Cuthill-McKee ordering

**Time estimate:** 7 hours

### Tasks
1. Implement `sparse_reorder_rcm()` in a new `src/sparse_reorder.c`:
   - Extract adjacency graph from input matrix
   - Find a pseudo-peripheral starting node (optional BFS-based heuristic for better results)
   - Run BFS from starting node, visiting neighbors in order of increasing degree
   - Reverse the resulting ordering (Cuthill-McKee → Reverse Cuthill-McKee)
   - Output: permutation array perm[] where perm[new_index] = old_index
2. Handle disconnected graphs:
   - If the graph has multiple connected components, process each separately
   - Concatenate the per-component orderings
3. Write unit tests:
   - Arrow matrix (dense first row/column) → RCM should produce banded structure
   - Already-banded matrix → RCM should not worsen bandwidth
   - Diagonal matrix → any permutation is valid
   - Disconnected matrix (block diagonal) → should handle without errors
   - Verify permutation is valid (contains each index exactly once)
4. Add `sparse_reorder.c` to build system
5. Run `make test` — all pass

### Deliverables
- Working `sparse_reorder_rcm()` implementation
- Handles disconnected graphs
- ≥4 RCM-specific tests
- Build system updated

### Completion Criteria
- RCM produces valid permutations for all test matrices
- Bandwidth is reduced (or not increased) on test cases
- `make test` and `make sanitize` clean

---

## Day 8: Fill-Reducing Reordering — RCM Validation & Bandwidth Metrics

**Theme:** Validate RCM on real matrices and add bandwidth measurement

**Time estimate:** 5 hours

### Tasks
1. Add bandwidth measurement utility:
   - `idx_t sparse_bandwidth(const SparseMatrix *A)` — compute max |i-j| over all nonzero a_ij
   - Useful for quantifying RCM effectiveness
2. Test RCM on all 6 SuiteSparse matrices:
   - Compute bandwidth before and after RCM reordering
   - Record fill-in after LU factorization with and without RCM
   - Identify which matrices benefit most from RCM
3. Add tests comparing bandwidth before/after RCM:
   - Assert bandwidth_after <= bandwidth_before for structured matrices
   - Verify factorization still produces correct results after reordering
4. Fix any bugs found during real-matrix validation
5. Run `make test` — all pass

### Deliverables
- `sparse_bandwidth()` utility function
- RCM bandwidth reduction data for all SuiteSparse matrices
- ≥3 validation tests on real matrices
- Bugs fixed (if any)

### Completion Criteria
- RCM reduces bandwidth on at least 3 of 6 SuiteSparse matrices
- LU factorization + solve produces correct results on RCM-reordered matrices
- `make test` and `make sanitize` clean

---

## Day 9: Fill-Reducing Reordering — AMD Implementation (Part 1)

**Theme:** Implement the core AMD algorithm

**Time estimate:** 8 hours

### Tasks
1. Implement `sparse_reorder_amd()` in `src/sparse_reorder.c`:
   - Build the elimination graph from the symmetrized sparsity pattern
   - Maintain degree lists (approximate external degree of each variable)
   - Main loop: select minimum-degree node, eliminate it, update degrees of neighbors
   - Use the quotient graph representation to avoid explicitly forming fill-in during the symbolic elimination
2. Implement degree update logic:
   - When a node is eliminated, its neighbors' degrees change
   - Use approximate external degree (cheaper than exact minimum degree)
   - Handle mass elimination: nodes with identical adjacency structure can be eliminated together (optional optimization)
3. Handle edge cases:
   - Empty matrix → identity permutation
   - Dense matrix → any elimination order has same fill
   - Already-eliminated nodes → skip in degree selection
4. Write basic correctness tests:
   - Small (4x4, 5x5) matrices where optimal ordering is known
   - Verify output is a valid permutation
   - Verify fill-in is not increased vs natural ordering
5. Run `make test` — all pass

### Deliverables
- Core AMD algorithm implementation
- Basic correctness tests
- Handles edge cases without crashes

### Completion Criteria
- AMD produces valid permutations on all test cases
- Fill-in with AMD ordering ≤ fill-in with natural ordering on test matrices
- `make test` clean

---

## Day 10: Fill-Reducing Reordering — AMD Validation & Optimization

**Theme:** Validate AMD on real matrices and optimize performance

**Time estimate:** 7 hours

### Tasks
1. Test AMD on all 6 SuiteSparse matrices:
   - Compare fill-in: natural ordering vs RCM vs AMD
   - Record factorization time with each ordering
   - Identify which matrices benefit most from AMD vs RCM
2. Profile AMD performance:
   - Measure AMD ordering time vs factorization time
   - Ensure AMD overhead is small relative to factorization savings
   - Optimize hot paths if needed (degree list operations, graph traversal)
3. Add comprehensive tests:
   - AMD on each SuiteSparse matrix → verify valid permutation and correct solve
   - Compare AMD and RCM fill-in quantitatively
   - Stress test: generate random sparse matrices of increasing size, verify AMD doesn't crash
4. Fix any bugs found during real-matrix validation
5. Run `make test` and `make sanitize` — all clean

### Deliverables
- AMD validated on all SuiteSparse matrices
- Comparison data: natural vs RCM vs AMD fill-in and timing
- ≥3 additional AMD validation tests
- Performance within acceptable bounds

### Completion Criteria
- AMD reduces fill-in vs natural ordering on at least 4 of 6 matrices
- AMD ordering time < 50% of factorization time on all matrices
- All tests pass under sanitizers

---

## Day 11: Factorization Integration — Options Struct & Reorder-Factor Pipeline

**Theme:** Integrate reordering into the factorization pipeline

**Time estimate:** 6 hours

### Tasks
1. Add factorization options struct:
   - `typedef struct { sparse_pivot_t pivot; sparse_reorder_t reorder; double tol; } sparse_lu_opts_t`
   - Add `sparse_lu_factor_opts()` that accepts the options struct
   - Keep existing `sparse_lu_factor()` as a convenience wrapper with defaults (no reorder, given pivot, default tolerance)
2. Implement reorder-factor pipeline in `sparse_lu_factor_opts()`:
   - If reorder != NONE: compute permutation, apply to matrix, then factor
   - Store the reordering permutation alongside the LU factors so that solve can undo it
   - Ensure solve correctly applies: reorder → factor → solve → un-reorder
3. Write integration tests:
   - Factor with SPARSE_REORDER_RCM + partial pivoting → solve → correct result
   - Factor with SPARSE_REORDER_AMD + complete pivoting → solve → correct result
   - Factor with SPARSE_REORDER_NONE → same behavior as existing `sparse_lu_factor()`
   - Verify solve results match between reordered and non-reordered factorizations
4. Run `make test` — all pass

### Deliverables
- `sparse_lu_opts_t` options struct
- `sparse_lu_factor_opts()` with reordering integration
- ≥4 integration tests
- Existing API backward-compatible

### Completion Criteria
- Reorder-factor-solve pipeline produces correct results
- `sparse_lu_factor()` still works unchanged
- All tests pass under sanitizers

---

## Day 12: Benchmarking — Reordering Effectiveness

**Theme:** Comprehensive benchmarking of reordering strategies

**Time estimate:** 5 hours

### Tasks
1. Update `bench_main.c` to support reordering options:
   - Add `--reorder none|rcm|amd` command-line flag
   - Report reordering time separately from factorization time
   - Include bandwidth before/after in tabular output
2. Run comprehensive benchmarks:
   - All 6 SuiteSparse matrices × 3 orderings (none, RCM, AMD) × 2 pivoting modes
   - Record: reorder time, factor time, solve time, fill-in ratio, residual, bandwidth
3. Analyze results:
   - Which ordering works best for each matrix type?
   - How much does reordering reduce fill-in on steam1 and orsirr_1 (the high-fill matrices)?
   - Is AMD's extra ordering cost justified by fill-in savings?
   - Are there matrices where reordering hurts (increases fill-in)?
4. Write `docs/planning/EPIC_1/SPRINT_3/benchmark_results.md` with findings and recommendations

### Deliverables
- Updated benchmark tool with reordering support
- Full benchmark results table (6 matrices × 3 orderings × 2 pivots)
- Analysis document with recommendations

### Completion Criteria
- Benchmark runs to completion without errors
- Results demonstrate fill-in reduction on high-fill matrices
- Analysis identifies best-practice ordering recommendations

---

## Day 13: Documentation, Edge Cases & Hardening

**Theme:** Harden all new code and update documentation

**Time estimate:** 5 hours

### Tasks
1. Edge-case hardening:
   - Rectangular matrices → reordering functions return `SPARSE_ERR_SHAPE` (LU requires square)
   - 1x1 matrix → reordering is a no-op
   - Matrix with no off-diagonal entries → permutation is identity
   - NULL inputs to all new functions → proper error codes
   - Very large degree nodes in AMD → verify no integer overflow in degree tracking
2. Update `docs/algorithm.md`:
   - Add section on condition number estimation (algorithm description, limitations)
   - Add section on fill-reducing reordering (RCM and AMD descriptions, when to use each)
   - Document the reorder-factor-solve pipeline
3. Update `README.md`:
   - Add condition estimation to feature list
   - Add reordering to feature list
   - Document `sparse_lu_opts_t` usage
4. Run full test suite and sanitizers
5. Run `make bench` and `make bench-suitesparse` — all clean

### Deliverables
- Edge-case tests for all new functions
- Updated algorithm documentation
- Updated README
- Clean sanitizer runs

### Completion Criteria
- All edge-case tests pass
- Documentation covers all new features
- `make test`, `make sanitize`, `make bench` all clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final validation, cleanup, and retrospective

**Time estimate:** 5 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make bench` — benchmarks run, no crashes
   - `make bench-suitesparse` — SuiteSparse benchmarks with all reorderings
2. Code review pass:
   - All new public API functions have doc comments
   - All new error codes handled in `sparse_strerror()`
   - `const` correctness on all new functions
   - No compiler warnings with strict flags
3. Verify backward compatibility:
   - Existing code using `sparse_lu_factor()` works unchanged
   - No new required fields or breaking API changes
4. Write `docs/planning/EPIC_1/SPRINT_3/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, assertion count, API functions, etc.)
   - Condition number data for SuiteSparse matrices
   - Reordering effectiveness summary
   - Items deferred to Sprint 4

### Deliverables
- All tests pass under all sanitizers
- Updated README with new API surface
- Sprint retrospective document
- Clean git history with meaningful commits

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 UBSan findings
- `make bench` and `make bench-suitesparse` complete without error
- README reflects current API
- Retrospective written with honest assessment

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | Condest — research & API design | 5 | API declaration, algorithm study, design notes |
| 2 | Condest — transpose solve | 6 | `sparse_lu_solve_transpose()`, ≥3 tests |
| 3 | Condest — Hager's algorithm | 6 | `sparse_lu_condest()` implementation, ≥4 tests |
| 4 | Condest — warnings & polish | 5 | Warning system, SuiteSparse validation, benchmark output |
| 5 | Reorder — research & data structures | 5 | API declarations, `sparse_reorder_t` enum, design notes |
| 6 | Reorder — graph extraction & permute | 6 | `sparse_permute()`, adjacency graph utility, ≥4 tests |
| 7 | Reorder — RCM implementation | 7 | `sparse_reorder_rcm()`, handles disconnected graphs, ≥4 tests |
| 8 | Reorder — RCM validation & bandwidth | 5 | `sparse_bandwidth()`, RCM data on SuiteSparse, ≥3 tests |
| 9 | Reorder — AMD implementation | 8 | Core AMD algorithm, basic tests |
| 10 | Reorder — AMD validation & optimization | 7 | AMD on SuiteSparse, comparison data, ≥3 tests |
| 11 | Integration — options struct & pipeline | 6 | `sparse_lu_opts_t`, reorder-factor-solve pipeline, ≥4 tests |
| 12 | Benchmarking — reordering effectiveness | 5 | Full benchmark table, analysis document |
| 13 | Documentation & hardening | 5 | Edge-case tests, algorithm docs, README update |
| 14 | Sprint review & retrospective | 5 | Retrospective, full regression, cleanup |

**Total estimate:** 81 hours (avg ~5.8 hrs/day, max 8 hrs/day)
