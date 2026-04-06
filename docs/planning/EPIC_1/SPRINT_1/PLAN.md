# Sprint 1 Plan: linalg_sparse_orthogonal Library

**Sprint Duration:** 14 days
**Goal:** Transform the prototype code into a proper C library with separated concerns, a build system, comprehensive tests, and bug fixes for all correctness issues identified in the initial review.

**Starting Point:** 7 monolithic `.c` files with duplicated code, no headers, no build system, no correctness tests, and several algorithmic bugs.

**End State:** A clean library with `include/`, `src/`, `tests/`, and `benchmarks/` directories; a CMake or Makefile build system; all correctness bugs fixed; comprehensive unit tests; performance benchmarks against known matrices.

---

## Day 1: Project Scaffolding & Consolidation

**Theme:** Establish project structure and single source of truth

### Tasks
1. Create directory structure:
   ```
   linalg_sparse_orthogonal/
   ├── include/           # Public headers
   ├── src/               # Library implementation
   ├── tests/             # Unit tests
   │   └── data/          # Test matrix files (.mtx)
   ├── benchmarks/        # Performance tests
   ├── archive/           # Old monolithic files (preserved)
   ├── planning/          # Plans and reviews (already exists)
   ├── CMakeLists.txt     # Build system
   ├── Makefile           # Simple Makefile alternative
   ├── .gitignore
   └── README.md
   ```
2. Move all 7 original `.c` files into `archive/` to preserve history
3. Initialize git repository with initial commit of the archive
4. Create `.gitignore` (build artifacts, `.o`, executables, `*.mtx` test outputs, etc.)
5. Create skeleton `CMakeLists.txt` with project definition, C11 standard, and warning flags

**Deliverable:** Clean directory structure, git repo initialized, originals preserved

---

## Day 2: Header Files & API Design

**Theme:** Define the public API and split declarations from implementations

### Tasks
1. Create `include/sparse_matrix.h`:
   - Opaque `typedef struct SparseMatrix SparseMatrix;`
   - Error code `enum` (replace `#define` constants with `typedef enum { ... } sparse_err_t;`)
   - Configuration constants (`DROP_TOL`, `NODES_PER_SLAB`) as adjustable `#define`s
   - Public API function prototypes with `const` correctness:
     - Lifecycle: `createSparseMatrix`, `freeSparseMatrix`, `copySparseMatrix`
     - Element access: `sparse_insert`, `sparse_remove`, `sparse_get`, `sparse_set` (physical and logical)
     - Info: `sparse_nnz`, `sparse_rows`, `sparse_cols`, `sparse_memory_usage`
     - I/O: `sparse_save_mm`, `sparse_load_mm`
     - Display: `sparse_print_dense`, `sparse_print_sparse`
   - Guard with `#ifndef SPARSE_MATRIX_H` / `#define` / `#endif`
2. Create `include/sparse_lu.h`:
   - LU decomposition: `sparse_lu_factor` (with pivoting strategy option)
   - Solve: `sparse_lu_solve`
   - Individual phases: `sparse_forward_sub`, `sparse_backward_sub`
   - Permutation helpers: `sparse_apply_row_perm`, `sparse_apply_inv_col_perm`
3. Create `include/sparse_types.h`:
   - `idx_t` typedef
   - `sparse_err_t` enum
   - Error code string conversion: `sparse_strerror(sparse_err_t)`
4. Decide on naming convention: **snake_case throughout** with `sparse_` prefix for public API

**Deliverable:** Three header files defining the complete public API

---

## Day 3: Core Library Implementation (Part 1 — Data Structure)

**Theme:** Implement the sparse matrix data structure in `src/`

### Tasks
1. Create `src/sparse_matrix.c`:
   - Include private struct definition (Node, NodeSlab, NodePool, SparseMatrix)
   - Implement pool allocator (`pool_alloc`, `pool_free_all`) as `static` functions
   - **Add free-list to pool:** Modified `Node` gets a union or reuse of `right` pointer for free-list chain; `removeNode` pushes onto free-list; `pool_alloc` pops from free-list first
   - Implement `createSparseMatrix` — accept `(idx_t rows, idx_t cols)` to support rectangular matrices
   - Implement `freeSparseMatrix`
   - Implement `copySparseMatrix` (deep copy — allocate new matrix, walk all row lists, insert each node)
2. Create `src/sparse_types.c`:
   - Implement `sparse_strerror()` — switch on error code, return string
3. Update `CMakeLists.txt`:
   - Add library target (`sparse_lu_ortho` static library)
   - Set include directories
   - Link `libm`

**Deliverable:** Compiles as a static library (data structure creation/destruction only)

---

## Day 4: Core Library Implementation (Part 2 — Insert/Remove/Access)

**Theme:** Implement all element access operations

### Tasks
1. In `src/sparse_matrix.c`, implement:
   - `sparse_insert(mat, row, col, val)` — physical index insertion with error codes
   - `sparse_remove(mat, row, col)` — physical index removal with free-list push
   - `sparse_get_phys(mat, row, col)` — physical index lookup
   - `sparse_get(mat, log_row, log_col)` — logical index lookup via permutations
   - `sparse_set(mat, log_row, log_col, val)` — logical index insert via permutations
   - `sparse_nnz(mat)`, `sparse_rows(mat)`, `sparse_cols(mat)` — info accessors
   - `sparse_memory_usage(mat)` — memory estimator (document as lower bound)
2. Add `const` qualifiers to all read-only parameters
3. Add bounds checking on all index parameters (return `SPARSE_ERR_INVALID_DIM`)
4. Verify the library compiles and links cleanly with `-Wall -Wextra -Werror`

**Deliverable:** Complete element access API, compiling with strict warnings

---

## Day 5: Core Library Implementation (Part 3 — I/O & Utilities)

**Theme:** Matrix Market I/O and utility functions

### Tasks
1. In `src/sparse_matrix.c` (or a new `src/sparse_io.c`), implement:
   - `sparse_save_mm(mat, filename)` — Matrix Market coordinate format output
   - `sparse_load_mm(mat_ptr, filename)` — Matrix Market coordinate format input
   - Handle symmetric matrices in MM format (read lower triangle, mirror to upper)
   - Handle pattern-only matrices (no values — use 1.0)
   - Validate header line more carefully (check "matrix", "coordinate", "real"/"pattern"/"integer")
2. Implement display functions:
   - `sparse_print_dense(mat, stream)` — O(n^2) dense-format print (with size guard: warn if n > 50)
   - `sparse_print_sparse(mat, stream)` — Print only non-zero entries as `(i, j, val)` triples
   - `sparse_print_info(mat, stream)` — Print dimensions, nnz, memory usage
3. Implement sparse matrix-vector multiply:
   - `sparse_matvec(mat, x, y)` — `y = A*x` using row-list traversal (O(nnz))

**Deliverable:** Full I/O and utility functions, library can load SuiteSparse matrices

---

## Day 6: LU Factorization (Bug Fixes & Reimplementation)

**Theme:** Fix the two critical correctness bugs and reimplement LU

### Tasks
1. Create `src/sparse_lu.c`, implement `sparse_lu_factor(mat, tol)`:
   - **Fix Bug 3.1:** Before the elimination inner loop, collect all physical row indices that need elimination into a temporary array (snapshot). Then iterate over the snapshot instead of walking `colHeaders[pk]->down` during modification.
   - **Fix Bug 3.3:** In `sparse_forward_sub`, remove the `break` on `j >= i`. Instead, traverse the entire row and accumulate only entries where `logical_col < i`. (Alternatively, skip entries with `j >= i` using `continue` instead of `break`.)
   - Clean up variable names: use descriptive names (`log_row`, `phys_col`, `pivot_log_row`, etc.)
   - Use `sparse_err_t` return codes throughout
2. Implement permutation swap helpers as static functions to reduce code duplication:
   - `swap_row_perm(mat, log_a, log_b)`
   - `swap_col_perm(mat, log_a, log_b)`
3. Implement `sparse_lu_solve(mat, b, x)`:
   - Allocates temporaries, chains: row_perm -> forward_sub -> backward_sub -> inv_col_perm
   - Proper cleanup on error (goto cleanup pattern)

**Deliverable:** Correct LU factorization and solve, all known bugs fixed

---

## Day 7: Unit Test Framework & Basic Tests

**Theme:** Build a proper test infrastructure and write data-structure tests

### Tasks
1. Create `tests/test_framework.h`:
   - Enhanced macro framework:
     - `TEST_SUITE_BEGIN(name)` / `TEST_SUITE_END()`
     - `TEST_CASE(name)` / `TEST_CASE_END()`
     - `ASSERT_TRUE(cond)`, `ASSERT_FALSE(cond)`
     - `ASSERT_EQ(a, b)`, `ASSERT_NEQ(a, b)`
     - `ASSERT_NEAR(a, b, tol)` — for floating-point comparison
     - `ASSERT_ERR(expr, expected_err)` — for error code checking
   - Test runner: auto-registration or explicit list
   - Summary with pass/fail counts and timing
2. Create `tests/test_sparse_matrix.c`:
   - **Test group: Creation**
     - Create matrix, verify dimensions, nnz == 0
     - Create with n=0 returns NULL
     - Create with n=1 works
   - **Test group: Insert/Get**
     - Insert single element, get it back
     - Insert multiple elements, get each back
     - Insert at same position overwrites
     - Insert 0.0 removes element
     - Get from empty position returns 0.0
   - **Test group: Remove**
     - Remove existing element, verify gone
     - Remove non-existent element, no error
     - NNZ tracks correctly through insert/remove
   - **Test group: Bounds checking**
     - Insert/get/set/remove with out-of-bounds indices
     - NULL pointer arguments
   - **Test group: Copy**
     - Copy matrix, verify all elements match
     - Modify copy, verify original unchanged
3. Update `CMakeLists.txt` — add test executable targets, `enable_testing()`, `add_test()`

**Deliverable:** Test framework, data structure tests, all passing

---

## Day 8: LU Correctness Tests

**Theme:** Comprehensive tests for LU factorization and solve

### Tasks
1. Create `tests/test_sparse_lu.c`:
   - **Test group: Known solutions**
     - 1x1 matrix: `[5]x = [10]` -> `x = [2]`
     - 2x2 matrix with known solution
     - 3x3 matrix from the original demo (verify `x` analytically)
     - 4x4 matrix with known solution
   - **Test group: Identity matrix**
     - LU of identity should produce L=I, U=I
     - Solve Ix = b should give x = b
   - **Test group: Diagonal matrix**
     - LU of diagonal matrix
     - Solve with diagonal matrix
   - **Test group: Triangular matrices**
     - Upper triangular: LU should be trivial (L=I)
     - Lower triangular: LU should produce correct factors
   - **Test group: Permutation verification**
     - After LU, verify `perm[inv_perm[i]] == i` for all i (both row and col)
     - Verify P*A*Q = L*U by explicit multiplication
   - **Test group: Residual checks**
     - Copy matrix before LU
     - Solve, compute `||Ax - b|| / (||A|| * ||x|| + ||b||)`
     - Assert relative residual < threshold (e.g., `n * eps`)
   - **Test group: Singular detection**
     - Zero matrix
     - Rank-deficient matrix
     - Nearly-singular matrix (condition number > 1e15)
   - **Test group: Drop tolerance**
     - Matrix where fill-in should be dropped
     - Verify nnz after factorization is less than full fill-in
2. Create helper function: `create_test_matrix(type, n)` — generates identity, diagonal, tridiagonal, random sparse, Hilbert, etc.

**Deliverable:** Comprehensive LU tests, all passing

---

## Day 9: Matrix Market I/O Tests & Test Matrices

**Theme:** Test I/O round-trips and add reference test matrices

### Tasks
1. Create `tests/test_sparse_io.c`:
   - **Test group: Save/Load round-trip**
     - Create matrix, save to .mtx, load back, compare all elements
     - Verify nnz matches after round-trip
   - **Test group: Error paths**
     - Load non-existent file
     - Load file with bad header
     - Load file with wrong dimensions
     - Save to invalid path
     - NULL pointer arguments
   - **Test group: Format edge cases**
     - Matrix with only one element
     - Empty matrix (0 nnz)
     - Very large values, very small values (precision preservation)
2. Add test matrices to `tests/data/`:
   - `identity_5.mtx` — 5x5 identity
   - `diagonal_10.mtx` — 10x10 diagonal
   - `tridiagonal_20.mtx` — 20x20 tridiagonal (Poisson-like)
   - `bcsstk01.mtx` — small structural engineering matrix from SuiteSparse (48x48, SPD)
   - `orsirr_1.mtx` or similar — small unsymmetric matrix from SuiteSparse
   - Write a small Python or C utility to generate standard test matrices if external download is not feasible
3. Create `tests/test_known_matrices.c`:
   - Load each test matrix, factorize, solve with known RHS, check residual
   - Report solve time and memory usage (informational, not pass/fail yet)

**Deliverable:** I/O tests passing, test matrices in repository, known-matrix tests passing

---

## Day 10: Sparse Matrix-Vector Product & Iterative Refinement

**Theme:** Add SpMV and use it for proper residual computation and iterative refinement

### Tasks
1. In `src/sparse_matrix.c`, verify/polish `sparse_matvec(mat, x, y)`:
   - Walk each physical row, accumulate `y[logical_row] += val * x[logical_col]`
   - Must use permutations correctly (operate in logical space)
   - Add `const` qualifiers
2. Add vector utility functions in `src/sparse_vector.c` (or inline in header):
   - `vec_norm2(v, n)` — 2-norm
   - `vec_norminf(v, n)` — infinity norm
   - `vec_axpy(a, x, y, n)` — `y += a*x`
   - `vec_copy(src, dst, n)`
3. Implement iterative refinement in `src/sparse_lu.c`:
   - `sparse_lu_refine(mat_orig, mat_lu, b, x, max_iters, tol)`:
     - Compute `r = b - A*x` using `sparse_matvec` on the *original* matrix
     - Solve `A*d = r` using the already-factored LU
     - Update `x += d`
     - Repeat until `||r|| < tol` or max iterations
4. Add tests in `tests/test_sparse_lu.c`:
   - Test `sparse_matvec` against dense computation
   - Test iterative refinement improves solution for ill-conditioned system

**Deliverable:** SpMV, vector utilities, iterative refinement, tested

---

## Day 11: Performance Benchmarks

**Theme:** Create a benchmarking harness and establish baseline performance

### Tasks
1. Create `benchmarks/bench_main.c`:
   - Command-line interface: `./bench [matrix.mtx] [--size N] [--repeat R]`
   - If `matrix.mtx` is provided, load it; otherwise, generate a random sparse matrix of size N
   - Time the following operations (using `clock_gettime` or `gettimeofday`):
     - Matrix construction (insert all elements)
     - LU factorization
     - Solve (single RHS)
     - Solve (multiple RHS)
     - SpMV
   - Report: wall-clock time, nnz before/after factorization, memory usage, residual norm
2. Create `benchmarks/bench_scaling.c`:
   - Generate tridiagonal matrices of sizes 100, 500, 1000, 2000, 5000
   - Measure factorization and solve time for each
   - Print results as a table (size, nnz, factor_time, solve_time, memory)
   - Verify O(n) scaling for tridiagonal (no fill-in)
3. Create `benchmarks/bench_fillin.c`:
   - Generate matrices with known fill-in patterns
   - Measure nnz growth during factorization
   - Report fill-in ratio (nnz_after / nnz_before)
4. Update `CMakeLists.txt` — add benchmark targets (not run by default `ctest`)

**Deliverable:** Benchmark harness, scaling tests, fill-in tests, baseline performance numbers

---

## Day 12: Documentation & README

**Theme:** Document the API, algorithms, and usage

### Tasks
1. Create `README.md`:
   - Project description and purpose
   - Build instructions (CMake and Makefile)
   - Quick start example (create matrix, factorize, solve)
   - API overview with links to headers
   - Performance characteristics and complexity notes
   - Known limitations
2. Add Doxygen-style comments to all public functions in headers:
   - Brief description
   - Parameter descriptions (including ownership semantics)
   - Return value description
   - Error conditions
   - Example usage (for key functions)
3. Create `docs/algorithm.md`:
   - Description of the orthogonal linked-list data structure
   - Complete pivoting LU algorithm pseudocode
   - Forward/backward substitution with permutations
   - Drop tolerance strategy
   - Complexity analysis (time and space)
4. Create `docs/matrix_market.md`:
   - Supported Matrix Market format features
   - Limitations (which subtypes are not supported)
   - Links to SuiteSparse Matrix Collection for test matrices

**Deliverable:** Complete documentation

---

## Day 13: Polish, Edge Cases & Hardening

**Theme:** Fix remaining issues, handle edge cases, clean up warnings

### Tasks
1. Audit all compiler warnings:
   - Compile with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`
   - Fix every warning
2. Run under address sanitizer:
   - Compile with `-fsanitize=address,undefined`
   - Run all tests, fix any issues
3. Run under valgrind (if available on platform):
   - Check for memory leaks
   - Check for use-after-free, buffer overflows
4. Edge case hardening:
   - 1x1 matrices
   - Matrices with a single non-zero element
   - Matrices with all entries on the diagonal
   - Matrix with nnz == 0 (all zeros)
   - Very large values (1e300) and very small values (1e-300)
   - Negative indices (should be caught by bounds checks)
5. Review and fix any remaining inconsistencies between the `_mm_full.c` reference and the new library implementation
6. Clean up the `archive/` directory — add a `README` explaining these are the original prototype files

**Deliverable:** Clean compile under strict settings, no sanitizer/valgrind errors, edge cases handled

---

## Day 14: Integration Testing, CI Setup & Sprint Review

**Theme:** End-to-end integration tests, wrap up, retrospective

### Tasks
1. Create `tests/test_integration.c` — end-to-end workflows:
   - Load Matrix Market file -> factorize -> solve -> check residual -> save result
   - Create matrix programmatically -> copy -> factorize copy -> solve -> refine -> verify
   - Multiple solves with same factorization (different RHS vectors)
   - Round-trip: create -> save -> load -> compare
2. Create a simple CI script (`scripts/ci.sh` or GitHub Actions `.github/workflows/ci.yml`):
   - Build library and all tests
   - Run all tests
   - Run benchmarks (informational only)
   - Report pass/fail
3. Final `README.md` updates:
   - Add build status badge placeholder
   - Add "Running Tests" section
   - Add "Benchmarks" section with sample output
4. Sprint retrospective:
   - Review all items from initial-review.md — mark each as addressed or deferred
   - Update `docs/planning/EPIC_1/SPRINT_1/RETROSPECTIVE.md` with:
     - What was completed
     - What was deferred to Sprint 2
     - Lessons learned
     - Sprint 2 candidate items (partial pivoting, threading, Cholesky, etc.)

**Deliverable:** Integration tests passing, CI script, sprint retrospective document

---

## Sprint 1 Definition of Done

All of the following must be true at the end of Day 14:

- [ ] Single authoritative library codebase in `src/` with headers in `include/`
- [ ] Old files preserved in `archive/`
- [ ] Build system compiles library, tests, and benchmarks
- [ ] All correctness bugs from initial review are fixed (Sections 3.1, 3.3, 3.6, 4.2)
- [ ] Error handling uses typed enum, all functions return error codes
- [ ] Pool allocator includes free-list for node reuse
- [ ] Comprehensive unit tests cover data structure, LU, I/O, and integration
- [ ] All tests pass under address sanitizer
- [ ] Performance benchmarks exist and produce results for tridiagonal scaling
- [ ] At least 3 external test matrices from SuiteSparse or equivalent
- [ ] README with build instructions and quick start
- [ ] API documentation in header files

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Free-list pool changes break correctness | Medium | High | Extensive testing before/after; sanitizer validation |
| Matrix Market format edge cases | Medium | Medium | Focus on "coordinate real general" subset; defer symmetric/complex |
| SuiteSparse matrix download issues | Low | Low | Generate equivalent test matrices programmatically |
| Forward-sub fix changes numerical results | Medium | Medium | Compare against dense LU reference implementation |
| Build system portability (macOS vs Linux) | Low | Medium | Test both; use standard C11 and POSIX APIs only |

---

## Daily Standup Template

Each day, before starting work:
1. What was completed yesterday?
2. What is planned for today?
3. Any blockers?

Track in `docs/planning/EPIC_1/SPRINT_1/DAILY_LOG.md` (create on Day 1).
