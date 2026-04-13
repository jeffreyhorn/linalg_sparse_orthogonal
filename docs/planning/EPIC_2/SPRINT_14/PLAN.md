# Sprint 14 Plan: Symbolic Analysis / Numeric Factorization Split

**Sprint Duration:** 14 days
**Goal:** Separate symbolic analysis from numeric factorization for LU, Cholesky, and LDL^T. This enables repeated numeric refactorization on the same sparsity pattern without redoing ordering and symbolic work — a critical capability for nonlinear solvers, time-stepping codes, and optimization loops that solve many systems with the same structure but different values.

**Starting Point:** A sparse linear algebra library with 16 headers, ~130 public API functions, 976 tests across 35 suites, LU/Cholesky/QR/SVD/LDL^T factorization, CG/GMRES/MINRES iterative solvers with ILU(0)/ILUT/IC(0) preconditioning, block solvers, matrix-free variants, AMD/RCM fill-reducing reordering, Bunch-Kaufman symmetric pivoting, norm-relative tolerance strategy, factored-state validation, and thread-safe norm caching. All factorizations currently perform ordering and symbolic analysis as part of each factor call.

**End State:** `sparse_analyze()` computes the elimination tree and symbolic factorization structure. `sparse_factor_numeric()` performs numeric-only factorization using a precomputed analysis. `sparse_refactor_numeric()` reuses symbolic structure for repeated factorization of matrices with the same sparsity pattern. Existing one-shot APIs (`sparse_lu_factor()`, `sparse_cholesky_factor()`, `sparse_ldlt_factor()`) remain backward compatible by internally calling analyze + factor. Benchmark showing speedup from symbolic-once approach.

---

## Day 1: Elimination Tree — Algorithm & Data Structures

**Theme:** Implement the elimination tree (etree) computation from a sparse matrix's sparsity pattern

**Time estimate:** 12 hours

### Tasks
1. Design `sparse_etree_t` internal struct in `src/sparse_analysis_internal.h`:
   - `idx_t *parent` — parent pointers (parent[i] = j means column j depends on column i)
   - `idx_t *postorder` — postorder traversal of the etree
   - `idx_t n` — matrix dimension
   - `idx_t *first_desc` — first descendant in postorder (for subtree range queries)
2. Implement `sparse_etree_compute(const SparseMatrix *A, idx_t *parent)`:
   - Liu's algorithm: for each column j, walk the lower triangular entries and use path compression to find the etree parent
   - O(nnz * alpha(n)) amortized via union-find with path compression
   - Handle symmetric matrices (use lower triangle only)
3. Implement `sparse_etree_postorder(const idx_t *parent, idx_t n, idx_t *postorder)`:
   - DFS-based postorder traversal of the etree
   - Needed for bottom-up symbolic factorization
4. Write initial tests in `tests/test_analysis.c`:
   - Diagonal matrix: etree has no edges (all roots)
   - Tridiagonal: etree is a path (parent[i] = i+1)
   - Arrow matrix: etree is a star (all point to last column)
   - Known 5x5 example with manually verified etree
5. Add to Makefile and CMakeLists.txt
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `src/sparse_analysis_internal.h` with etree data structures
- Etree computation via Liu's algorithm with path compression
- Postorder traversal
- Basic etree tests

### Completion Criteria
- Etree is correct for diagonal, tridiagonal, arrow, and known matrices
- Postorder is a valid DFS ordering of the etree
- `make format && make lint && make test` clean

---

## Day 2: Elimination Tree — Hardening & Column Counts

**Theme:** Compute symbolic column counts from the etree (nonzero count per column of L)

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_colcount(const SparseMatrix *A, const idx_t *parent, const idx_t *postorder, idx_t *colcount)`:
   - Compute the exact number of nonzeros in each column of L using the etree
   - Use the skeleton-matrix algorithm (Gilbert, Ng, Peyton): for each column j, count the number of rows in the subtree rooted at j that have entries in column j
   - O(nnz) time with the etree and postorder
2. Add column count validation tests:
   - Diagonal matrix: colcount[i] = 1 (diagonal only)
   - Tridiagonal SPD: colcount[i] = 2 (diagonal + one subdiagonal) except last = 1
   - Dense lower triangle: colcount[i] = n-i
   - Known example with manually verified column counts
3. Test that sum of column counts equals nnz(L) for Cholesky on test matrices
4. Edge cases:
   - 1x1 matrix
   - Empty columns (structurally zero columns)
   - Symmetric permuted matrix (verify etree accounts for permutation)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_colcount()` implementation
- Column count tests with known results
- Verified nnz(L) prediction matches actual Cholesky nnz

### Completion Criteria
- Column counts match actual Cholesky factor column lengths
- O(nnz) algorithm performance
- `make format && make lint && make test` clean

---

## Day 3: Symbolic Cholesky Factorization

**Theme:** Compute the complete symbolic structure of L (row indices per column) without numeric work

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_symbolic_cholesky(const SparseMatrix *A, const idx_t *parent, const idx_t *postorder, const idx_t *colcount, sparse_symbolic_t *sym)`:
   - Allocate row index arrays for each column of L using the precomputed column counts
   - Compute the actual row indices using the etree: for each column j in postorder, merge the row index sets of children and add the original entries from A
   - Store as compressed column format: `sym->col_ptr`, `sym->row_idx`, `sym->nnz`
2. Design `sparse_symbolic_t` struct:
   - `idx_t *col_ptr` — column pointers (length n+1)
   - `idx_t *row_idx` — row indices (length nnz)
   - `idx_t n` — dimension
   - `idx_t nnz` — total nonzeros in L
3. Write symbolic Cholesky tests:
   - Compare symbolic structure with actual Cholesky: every nonzero in L should appear in the symbolic structure
   - Verify no extra entries (symbolic nnz == numeric nnz for Cholesky)
   - Test on SuiteSparse bcsstk04 and nos4
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_symbolic_cholesky()` producing exact L structure
- `sparse_symbolic_t` struct with compressed column storage
- Tests verifying symbolic matches numeric Cholesky structure

### Completion Criteria
- Symbolic nnz(L) exactly matches numeric Cholesky nnz(L)
- Row indices per column match actual Cholesky L
- `make format && make lint && make test` clean

---

## Day 4: Symbolic LU Factorization

**Theme:** Compute upper bounds on L and U sparsity structure for LU factorization

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_symbolic_lu(const SparseMatrix *A, const idx_t *perm, sparse_symbolic_t *sym_L, sparse_symbolic_t *sym_U)`:
   - Use the column etree of A (or of P*A for pivoted LU) to compute upper bounds on L and U structure
   - For partial pivoting, the symbolic structure is an upper bound (actual fill may be less due to cancellation)
   - Pre-allocate L and U storage based on the upper bound
2. Handle permutation:
   - If AMD/RCM permutation is provided, compute etree of permuted matrix
   - Store the permutation in the analysis object for use during numeric factorization
3. Write symbolic LU tests:
   - Compare predicted nnz with actual LU nnz (predicted >= actual)
   - Verify all actual L/U nonzero positions are in the predicted structure
   - Test with and without AMD reordering
   - Test on unsymmetric matrices
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_symbolic_lu()` producing upper-bound L/U structure
- Permutation-aware etree computation
- Tests verifying symbolic structure contains all numeric nonzeros

### Completion Criteria
- Symbolic nnz >= numeric nnz for all test matrices
- All numeric nonzero positions appear in symbolic structure
- AMD-reordered symbolic factorization works correctly
- `make format && make lint && make test` clean

---

## Day 5: Analysis Object API Design

**Theme:** Design the public `sparse_analysis_t` struct and `sparse_analyze()` API

**Time estimate:** 10 hours

### Tasks
1. Design `sparse_analysis_t` public struct in new `include/sparse_analysis.h`:
   - `idx_t n` — matrix dimension
   - `idx_t *perm` — fill-reducing permutation (or NULL for natural order)
   - `idx_t *etree` — elimination tree parent pointers
   - `idx_t *postorder` — etree postorder
   - `sparse_symbolic_t sym_L` — symbolic structure of L
   - `sparse_symbolic_t sym_U` — symbolic structure of U (NULL for Cholesky/LDL^T)
   - `sparse_factor_type_t type` — CHOLESKY, LU, or LDLT
   - `double analysis_norm` — cached ||A||_inf for tolerance
2. Design analysis options:
   - `sparse_analysis_opts_t` with `reorder` (NONE/RCM/AMD), `factor_type` (CHOLESKY/LU/LDLT)
3. Design public API:
   - `sparse_analyze(const SparseMatrix *A, const sparse_analysis_opts_t *opts, sparse_analysis_t *analysis)` — compute symbolic analysis
   - `sparse_analysis_free(sparse_analysis_t *analysis)` — free analysis data
4. Write header with full Doxygen documentation
5. Create stub `src/sparse_analysis.c` with function signatures
6. Add to build system
7. Run `make format && make lint` — all clean

### Deliverables
- `include/sparse_analysis.h` with complete API documentation
- `src/sparse_analysis.c` stub
- Build system updated

### Completion Criteria
- `make clean && make` builds with new files
- Headers self-documenting with `@pre`, `@note`, return codes
- `make format && make lint` clean

---

## Day 6: sparse_analyze() Implementation — Cholesky Path

**Theme:** Implement `sparse_analyze()` for the Cholesky factorization path

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_analyze()` for `SPARSE_FACTOR_CHOLESKY`:
   - Validate input (NULL, square, symmetric)
   - Apply fill-reducing reordering (AMD/RCM) if requested
   - Compute etree of (permuted) A
   - Compute postorder
   - Compute column counts
   - Compute symbolic Cholesky structure
   - Store everything in `sparse_analysis_t`
2. Implement `sparse_analysis_free()`:
   - Free all allocated arrays (perm, etree, postorder, sym_L, sym_U)
   - Safe on zeroed struct
3. Write analysis tests:
   - Analyze a tridiagonal SPD matrix: verify etree, column counts, symbolic structure
   - Analyze with AMD reordering: verify permutation stored correctly
   - Analyze bcsstk04: verify predicted nnz matches Cholesky nnz
   - Free and re-analyze: no leaks
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Working `sparse_analyze()` for Cholesky path
- `sparse_analysis_free()` implemented
- Analysis correctness tests

### Completion Criteria
- Analysis produces correct etree, column counts, and symbolic structure
- Predicted nnz(L) matches actual Cholesky nnz(L) on all test matrices
- `make format && make lint && make test` clean

---

## Day 7: sparse_analyze() — LU and LDL^T Paths

**Theme:** Extend `sparse_analyze()` to support LU and LDL^T factorization types

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_analyze()` for `SPARSE_FACTOR_LU`:
   - Compute etree of A^T * A (or column etree) for unsymmetric LU
   - Compute symbolic L and U upper bounds
   - Store both sym_L and sym_U in the analysis object
2. Implement `sparse_analyze()` for `SPARSE_FACTOR_LDLT`:
   - Use symmetric etree (same as Cholesky)
   - Symbolic structure of L is the same as Cholesky
   - Note: LDL^T with Bunch-Kaufman may have slightly different fill due to 2x2 pivoting, but the symbolic upper bound is valid
3. Write tests:
   - LU analysis on unsymmetric matrices: verify structure contains all numeric nonzeros
   - LDL^T analysis on KKT matrices: verify structure
   - Compare LU analysis nnz with actual LU nnz
   - Analysis type mismatch test: analyze as Cholesky, factor as LU → should fail
4. Run `make format && make lint && make test` — all clean

### Deliverables
- LU and LDL^T analysis paths
- Tests for all three factorization types
- Type-mismatch validation

### Completion Criteria
- LU symbolic structure contains all numeric LU nonzeros
- LDL^T symbolic structure contains all numeric LDL^T nonzeros
- `make format && make lint && make test` clean

---

## Day 8: Numeric Factorization with Precomputed Analysis — Cholesky

**Theme:** Implement `sparse_factor_numeric()` for Cholesky using precomputed symbolic structure

**Time estimate:** 12 hours

### Tasks
1. Design `sparse_factor_numeric()` API:
   - `sparse_factor_numeric(const SparseMatrix *A, const sparse_analysis_t *analysis, sparse_factors_t *factors)` — perform numeric-only factorization
   - `sparse_factors_t` wraps the factorization result (L for Cholesky, L+U for LU, L+D for LDL^T)
2. Implement Cholesky numeric factorization using symbolic structure:
   - Apply the stored permutation to A
   - Iterate columns in postorder
   - For each column, use the precomputed row indices from sym_L
   - Compute numeric values using the standard left-looking algorithm
   - Store directly into pre-allocated compressed column storage
3. Write tests:
   - Factor tridiagonal SPD: compare result with one-shot `sparse_cholesky_factor()`
   - Factor bcsstk04 with analysis: verify solve residual matches one-shot
   - Factor nos4: same comparison
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_factor_numeric()` for Cholesky
- `sparse_factors_t` result struct
- Numeric Cholesky tests comparing with one-shot API

### Completion Criteria
- Analyze+factor produces identical results to one-shot `sparse_cholesky_factor()`
- Solve residual matches one-shot on all test matrices
- `make format && make lint && make test` clean

---

## Day 9: Numeric Factorization — LU and LDL^T Paths

**Theme:** Extend `sparse_factor_numeric()` to LU and LDL^T

**Time estimate:** 12 hours

### Tasks
1. Implement LU numeric factorization using symbolic structure:
   - Apply stored permutation
   - Use pre-allocated L and U storage from analysis
   - Perform numeric elimination within the symbolic structure
   - Handle partial pivoting within the pre-allocated structure (pivoting may use fewer entries than the upper bound)
2. Implement LDL^T numeric factorization:
   - Apply stored permutation
   - Perform Bunch-Kaufman elimination within symbolic L structure
   - Store D diagonal and 2x2 block off-diagonals
3. Write tests:
   - LU: analyze+factor vs one-shot on unsymmetric matrices
   - LDL^T: analyze+factor vs one-shot on KKT matrices
   - Solve and verify residual for both paths
4. Run `make format && make lint && make test` — all clean

### Deliverables
- LU and LDL^T numeric factorization paths
- Tests comparing with one-shot APIs

### Completion Criteria
- Analyze+factor matches one-shot for LU and LDL^T
- Solve residuals match on all test matrices
- `make format && make lint && make test` clean

---

## Day 10: Numeric Refactorization

**Theme:** Implement `sparse_refactor_numeric()` for repeated factorization on the same pattern

**Time estimate:** 12 hours

### Tasks
1. Implement `sparse_refactor_numeric(const SparseMatrix *A_new, const sparse_analysis_t *analysis, sparse_factors_t *factors)`:
   - Validate that A_new has the same sparsity pattern as the original analysis matrix
   - Reuse the symbolic structure (no re-analysis)
   - Perform numeric factorization with the new values
   - Overwrite the existing numeric factors in-place (no reallocation)
2. Pattern validation:
   - Check dimension match
   - Check nnz match
   - Optionally check row/column index match (controlled by a flag)
3. Write refactorization tests:
   - Factor A, then refactor with A' (same pattern, different values): verify both solve correctly
   - Factor A, modify one value, refactor: verify updated solution
   - Factor A, add a new nonzero: refactor should fail (pattern mismatch)
   - Repeated refactorization in a loop (10 iterations): verify no memory growth
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_refactor_numeric()` with pattern validation
- Repeated refactorization tests
- Pattern mismatch detection

### Completion Criteria
- Refactorization produces correct results with new values
- Pattern mismatch detected and reported
- No memory leaks on repeated refactorization
- `make format && make lint && make test` clean

---

## Day 11: Backward Compatibility — One-Shot API Wrappers

**Theme:** Make existing one-shot APIs internally use analyze + factor, preserving backward compatibility

**Time estimate:** 10 hours

### Tasks
1. Refactor `sparse_cholesky_factor()` to internally:
   - Call `sparse_analyze(A, &opts, &analysis)` with Cholesky type
   - Call `sparse_factor_numeric(A, &analysis, &factors)`
   - Convert factors back to the existing `SparseMatrix *` format
   - Free the analysis object
   - No API change — existing callers see identical behavior
2. Refactor `sparse_lu_factor()` similarly
3. Refactor `sparse_ldlt_factor()` similarly
4. Run full regression:
   - All existing tests must pass unchanged
   - No behavior differences
5. Run `make format && make lint && make test` — all clean

### Deliverables
- One-shot APIs internally use analyze + factor
- Full backward compatibility (no test changes needed)

### Completion Criteria
- All 976+ existing tests pass without modification
- One-shot API behavior is identical
- `make format && make lint && make test` clean

---

## Day 12: Documentation & Example Program

**Theme:** Document the analysis/factorization API and create examples

**Time estimate:** 8 hours

### Tasks
1. Update README:
   - Add symbolic analysis to feature list
   - Add `sparse_analysis.h` to API overview
   - Add analyze/factor/refactor functions to key functions
   - Update project structure
2. Update `docs/algorithm.md`:
   - Add elimination tree algorithm description
   - Add symbolic factorization algorithm
   - Document the analyze → factor → refactor workflow
3. Create `examples/example_analysis.c`:
   - Demonstrate: analyze once, factor, solve, change values, refactor, solve again
   - Show speedup from skipping repeated symbolic analysis
4. Add example to CMakeLists.txt
5. Run `make format && make lint && make test` — all clean

### Deliverables
- README and algorithm docs updated
- Working example program
- Build system updated

### Completion Criteria
- Example compiles, runs, and demonstrates the analyze/factor/refactor workflow
- Documentation accurately describes the new API
- `make format && make lint && make test` clean

---

## Day 13: Full Regression & Benchmarks

**Theme:** Full regression, sanitizers, and performance benchmarks

**Time estimate:** 8 hours

### Tasks
1. Full regression:
   - `make clean && make test` — all tests pass
   - `make sanitize` — ASan/UBSan clean
   - `make bench` — benchmarks run without crashes
   - CMake: `ctest` all pass
2. Benchmark: symbolic-once vs repeated full factorization:
   - Factor the same matrix 100 times with full one-shot API
   - Factor once with analyze, then refactor 99 times
   - Measure and report speedup
   - Test on bcsstk04 and nos4
3. Memory benchmark:
   - Verify refactorization does not allocate additional memory after first factor
   - Compare peak memory: one-shot vs analyze+factor
4. Run `make format && make lint && make test` — final clean build

### Deliverables
- Full regression pass
- Benchmark data: symbolic-once speedup
- Memory usage comparison

### Completion Criteria
- All tests pass, sanitizers clean
- Benchmark shows measurable speedup from symbolic reuse
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final documentation, sprint review, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Final metrics collection:
   - Total test count
   - Analysis-specific test count
   - Benchmark: analyze-once speedup on reference matrices
   - Memory comparison data
2. Write `docs/planning/EPIC_2/SPRINT_14/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics
   - Items deferred (if any)
3. Update project plan if any Sprint 14 items were deferred
4. Run `make format && make lint && make test` — final clean build

### Deliverables
- Sprint retrospective document
- Updated metrics
- Clean final build

### Completion Criteria
- All Sprint 14 items complete or explicitly deferred
- Retrospective written with honest assessment
- `make format && make lint && make test` clean
