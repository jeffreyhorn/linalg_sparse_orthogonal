# Sprint 1 Daily Log

## Day 1 — Project Scaffolding & Consolidation

### Completed
- Created directory structure: `include/`, `src/`, `tests/`, `tests/data/`, `benchmarks/`, `archive/`, `scripts/`, `docs/`
- Moved all 7 original `.c` files and the compiled binary to `archive/`
- Added `archive/README.md` documenting the file evolution and known issues
- Initialized git repository
- Created `.gitignore`
- Created `CMakeLists.txt` with C11, strict warnings, optional sanitizers, library target, test/bench scaffolding
- Created `Makefile` as a simpler build alternative
- Created public headers:
  - `include/sparse_types.h` — `idx_t`, `sparse_err_t` enum, `sparse_strerror()`
  - `include/sparse_matrix.h` — full public API for sparse matrix data structure
  - `include/sparse_lu.h` — LU factorization and solve API
- Created private internal header:
  - `src/sparse_matrix_internal.h` — Node, NodeSlab, NodePool (with free-list), SparseMatrix struct
- Created implementation files:
  - `src/sparse_types.c` — error code to string
  - `src/sparse_matrix.c` — full implementation: pool with free-list, create/free/copy, insert/remove/get/set, matvec, Matrix Market I/O, display, permutation access
  - `src/sparse_lu.c` — LU factor (with snapshot fix for bug 3.1, forward-sub fix for bug 3.3), solve, individual phases, iterative refinement

### Notes
- Went beyond the original Day 1 plan to also complete Days 2-5 implementation work, since the API design and implementation flowed naturally from the consolidation
- All three critical bugs from the review are fixed in the new code:
  - Bug 3.1: `sparse_lu_factor` snapshots elimination row indices before modifying the matrix
  - Bug 3.3: `sparse_forward_sub` traverses the entire row without early break
  - Bug 3.6: `sparse_matvec` is provided so callers can compute proper residuals on the original matrix
- Pool allocator now includes a free-list for node reuse
- API uses `const` correctness and `sparse_err_t` throughout
- Rectangular matrices are supported in the data structure (`sparse_create(rows, cols)`)

## Day 2 — Header Files & API Design (Review & Polish)

### Completed
- Reviewed all 3 headers and 3 implementation files against the Day 2 plan
- Added `sparse_pivot_t` enum (`SPARSE_PIVOT_COMPLETE`, `SPARSE_PIVOT_PARTIAL`) to `sparse_types.h`
- Updated `sparse_lu_factor` signature to accept a pivoting strategy parameter
- Implemented partial pivoting in `sparse_lu.c` (searches only the pivot column)
- Added `SPARSE_ERR_SHAPE` error code for non-square matrix passed to LU
- Moved tuning constants to public header as overridable defines:
  - `SPARSE_NODES_PER_SLAB` (default 4096) — pool slab size
  - `SPARSE_DROP_TOL` (default 1e-14) — fill-in drop tolerance
- Internal header now derives `NODES_PER_SLAB` / `DROP_TOL` from public defines
- Added symmetric and pattern-only Matrix Market format support in `sparse_load_mm`
- Created `tests/smoke_test.c` — end-to-end verification:
  - Builds, links, and runs against the library
  - Tests complete pivoting, partial pivoting, copy, solve, residual, iterative refinement, and MM round-trip
  - All pass with zero residual
- Updated Makefile with generic test build rule and `make smoke` target

### Notes
- Most Day 2 work (headers, API design) was already done in Day 1
- Day 2 focused on reviewing for gaps and adding missing features from the plan
- Both pivoting strategies produce the correct solution for the 3x3 test matrix
- Days 3-6 from the original plan are also substantially complete

## Day 3 — Unit Test Framework & Test Suites

### Completed
- Created `tests/test_framework.h` — minimal macro-based test framework:
  - `TEST_SUITE_BEGIN`/`TEST_SUITE_END` with summary and timing
  - `RUN_TEST` with pass/fail reporting
  - Assertion macros: `ASSERT_TRUE`, `ASSERT_FALSE`, `ASSERT_EQ`, `ASSERT_NEQ`, `ASSERT_NEAR`, `ASSERT_ERR`, `ASSERT_NULL`, `ASSERT_NOT_NULL`
  - Reports all failures per test (does not stop at first assertion failure)
  - Returns exit code 1 if any test failed
- Created `tests/test_sparse_matrix.c` — 31 data structure tests:
  - Creation (basic, rectangular, 1x1, invalid args, free NULL)
  - Insert/get (single, multiple, overwrite, zero removes, empty matrix, row/col ordering)
  - Logical get/set through permutations
  - Remove (existing, nonexistent, middle of row, nnz tracking)
  - Bounds checking (insert, remove, set, get out of bounds, null pointers)
  - Copy (basic, independent modification, null)
  - Matrix-vector product (identity, general, null args)
  - Permutations (initial identity, reset)
  - Memory usage
- Created `tests/test_sparse_lu.c` — 24 LU factorization tests:
  - Known solutions (1x1, 2x2, 3x3, 4x4)
  - Special matrices (identity, diagonal, upper/lower triangular)
  - Tridiagonal (Poisson 1D, n=20)
  - Permutation consistency (complete and partial pivoting)
  - Residual checks on 10x10 diag-dominant matrix
  - Both pivoting strategies agree
  - Singular detection (zero matrix, rank deficient, zero row)
  - Error paths (null matrix, nonsquare, null solve/sub args)
  - Iterative refinement
  - Drop tolerance fill-in control
- Updated Makefile: uncommented `TEST_SRCS`/`TEST_BINS` for test_sparse_matrix and test_sparse_lu
- Updated CMakeLists.txt: uncommented `add_sparse_test` for both test suites
- **Found and fixed bug**: `sparse_apply_inv_col_perm` used `col_perm[i]` instead of `inv_col_perm[i]`, computing Q^{-1}*z instead of Q*z during the column back-permutation step of the solve. This caused incorrect solutions when complete pivoting produced non-self-inverse column permutations (e.g., the n=20 tridiagonal)

### Test Results
- `test_sparse_matrix`: 31/31 passed, 166 assertions
- `test_sparse_lu`: 24/24 passed, 84 assertions
- `smoke_test`: passed
- **Total: 55 unit tests, 250 assertions, all passing**

### Notes
- The column permutation bug was latent — small test matrices happened to produce self-inverse permutations (simple swaps), masking the error
- The tridiagonal n=20 test exposed the bug because complete pivoting produced a complex permutation
- This is the fourth major bug found (after bugs 3.1, 3.3, 3.6 from the initial review)

## Day 4 — Matrix Market I/O Tests & Reference Test Matrices

### Completed
- Created 8 reference test matrices in `tests/data/`:
  - `identity_5.mtx` — 5x5 identity
  - `diagonal_10.mtx` — 10x10 diagonal (d[i] = i+1)
  - `tridiagonal_20.mtx` — 20x20 Poisson-1D tridiagonal (-1, 2, -1)
  - `symmetric_4.mtx` — 4x4 symmetric (tests MM symmetric mirroring)
  - `pattern_3.mtx` — 3x3 pattern-only (tests MM pattern format)
  - `bcsstk01.mtx` — 6x6 SPD structural engineering matrix (two 3x3 blocks, inspired by BCS collection)
  - `unsymm_5.mtx` — 5x5 unsymmetric diag-dominant matrix
  - `bad_header.mtx` — deliberately invalid file for error testing
- Created `tests/test_sparse_io.c` — 18 I/O tests:
  - Round-trip: basic, single element, rectangular, precision (1e300 / 1e-300), nnz preservation
  - Loading test data: identity, diagonal, symmetric (with mirroring), pattern (values = 1.0)
  - Error paths: nonexistent file, bad header, null args (save and load), invalid path
  - Edge cases: 1x1 matrix, empty matrix (nnz=0), negative values, round-trip after LU permutation
- Created `tests/test_known_matrices.c` — 15 tests loading reference matrices:
  - Identity (complete + partial pivot)
  - Diagonal (complete + partial)
  - Tridiagonal (complete + partial + refined)
  - Symmetric (complete + partial)
  - bcsstk01-inspired (complete + refined)
  - Unsymmetric (complete + partial)
  - Solution accuracy checks (compare x against known x_exact = [1,...,1])
- Updated Makefile and CMakeLists.txt for new test targets

### Test Results
- `test_sparse_matrix`: 31/31 passed, 166 assertions
- `test_sparse_lu`: 24/24 passed, 84 assertions
- `test_sparse_io`: 18/18 passed, 142 assertions
- `test_known_matrices`: 15/15 passed, 41 assertions
- **Total: 88 unit tests, 433 assertions, all passing**

### Notes
- Caught an off-by-one in the hand-crafted bcsstk01.mtx (header said 13 nnz but file had 12 entries)
- `%.15g` format in `sparse_save_mm` preserves double-precision values through round-trip
- Both pivoting strategies produce correct results for all reference matrices
- The bcsstk01-inspired matrix has entries of order 1e6, validating that the solver handles diverse magnitudes

## Day 5 — Vector Utilities, SpMV Tests & Iterative Refinement Tests

### Completed
- Created `include/sparse_vector.h` and `src/sparse_vector.c` — dense vector utility module:
  - `vec_norm2(v, n)` — 2-norm
  - `vec_norminf(v, n)` — infinity-norm
  - `vec_axpy(a, x, y, n)` — y += a*x
  - `vec_copy(src, dst, n)` — copy
  - `vec_zero(v, n)` — zero fill
  - `vec_dot(x, y, n)` — dot product
  - All functions are NULL-safe (no-op on NULL inputs)
- Created `tests/test_sparse_vector.c` — 24 tests:
  - Vector utilities: norm2, norminf, axpy, copy, zero, dot (basic, edge cases, null safety)
  - SpMV: comparison vs dense matvec, rectangular matrix, empty matrix, extreme values (1e15 / 1e-15)
  - Iterative refinement: ill-conditioned 8x8 Hilbert-like system, zero RHS, multiple RHS with same factorization
- Updated Makefile and CMakeLists.txt: added sparse_vector.c to library sources, test_sparse_vector to test targets

### Test Results
- `test_sparse_matrix`: 31/31 passed, 166 assertions
- `test_sparse_lu`: 24/24 passed, 84 assertions
- `test_sparse_io`: 18/18 passed, 142 assertions
- `test_known_matrices`: 15/15 passed, 41 assertions
- `test_sparse_vector`: 24/24 passed, 52 assertions
- **Total: 112 unit tests, 485 assertions, all passing**

### Notes
- SpMV and iterative refinement were already implemented; this day added the vector utility module and comprehensive tests
- The ill-conditioned test verifies that refinement does not degrade the solution and achieves residual < 1e-12
- Multiple-RHS test validates that the same LU factorization can be reused for different right-hand sides

## Day 6 — Performance Benchmarks

### Completed
- Created `benchmarks/bench_main.c` — general-purpose benchmark harness:
  - CLI: `./bench_main [matrix.mtx]`, `--size N`, `--repeat R`
  - Generates diag-dominant random sparse if no file provided
  - Times SpMV, LU factorization, LU solve
  - Reports nnz, fill-in ratio, memory, residual norm
- Created `benchmarks/bench_scaling.c` — scaling analysis:
  - Tridiagonal with partial and complete pivoting (n = 100..5000)
  - Dense random with complete pivoting (n = 10..200)
  - Tabular output: n, nnz, nnz_LU, factor/solve/spmv time, memory, residual
- Created `benchmarks/bench_fillin.c` — fill-in analysis:
  - 5 matrix types: tridiagonal, pentadiagonal, arrow, random sparse, dense
  - Both pivoting strategies
  - Reports nnz before/after, density percentage, fill-in ratio
- Updated Makefile and CMakeLists.txt for benchmark targets (`make bench`)

### Benchmark Highlights
- **Tridiagonal (partial pivot)**: zero fill-in (ratio 1.00), linear scaling confirmed
  - n=5000: factor 0.44ms, solve 0.20ms
- **Tridiagonal (complete pivot)**: modest fill-in (ratio ~1.6) due to column pivoting
  - n=5000: factor 256ms (much slower due to submatrix search)
- **Arrow matrix**: catastrophic fill-in (100% dense after factorization) — expected
- **Dense n=200**: factor 1.57s, residual 1.8e-15 (near machine epsilon)
- **Partial pivoting is ~100-500x faster than complete pivoting** for structured matrices

### Notes
- All 112 unit tests still pass
- Benchmarks are informational (not part of `make test`)
- The O(n^3) complete pivoting search dominates for larger matrices; partial pivoting is strongly preferred for banded/structured matrices

## Day 7 — Documentation & README

### Completed
- Created `README.md`:
  - Project description and features
  - Build instructions (Make and CMake)
  - Quick start example (create, factor, solve, refine)
  - API overview table linking to all four headers
  - Key function summary
  - Performance characteristics table with benchmark data
  - Complexity notes
  - Known limitations section
  - Testing summary (112 tests, 485 assertions)
  - Project structure diagram
- Added Doxygen-style comments to all public functions in all four headers:
  - `sparse_types.h` — file brief, enum documentation, `sparse_strerror`
  - `sparse_matrix.h` — file brief, tuning constants, all 22 function prototypes documented with `@brief`, `@param`, `@return`, `@note`
  - `sparse_lu.h` — file brief, usage example in `@code` block, all 8 function prototypes documented
  - `sparse_vector.h` — file brief, NULL-safety noted, all 6 function prototypes documented
- Created `docs/algorithm.md`:
  - Orthogonal linked-list data structure description with Node struct
  - Advantages and trade-offs vs CSR/CSC
  - Slab pool allocator with free-list mechanism
  - Permutation array documentation (4 arrays, invariants)
  - LU factorization pseudocode with pivot selection
  - Snapshot mechanism explanation (Bug 3.1 fix)
  - Forward substitution fix explanation (Bug 3.3 fix)
  - Full solve procedure (4-step chain)
  - Iterative refinement algorithm
  - Complexity analysis tables (space and time)
  - Fill-in behavior table by matrix type
  - Drop tolerance discussion
- Created `docs/matrix_market.md`:
  - Supported read/write features table
  - Symmetry handling (automatic mirroring)
  - Pattern matrix handling (value = 1.0)
  - Unsupported features (array, complex, skew-symmetric, Hermitian)
  - File format reference with examples
  - Reference test matrix table
  - SuiteSparse usage guide

### Test Results
- All 112 tests pass, 485 assertions — no regressions from header documentation changes

### Notes
- Headers now serve as comprehensive API documentation
- The algorithm doc covers all four bugs found during the project and their fixes
- README performance table uses actual benchmark data from Day 6

## Day 8 — Polish, Edge Cases & Hardening

### Completed
- **Compiler warning audit**: Zero warnings across library, all 6 test suites, and all 3 benchmarks with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`
- **UBSan testing**: All tests pass under `-fsanitize=undefined` with zero undefined behavior detected
  - ASan (`-fsanitize=address`) hangs on this macOS/sandbox environment even for trivial programs — this is a platform limitation, not a code issue
  - Updated Makefile `sanitize` target to use UBSan only
- Created `tests/test_edge_cases.c` — 20 edge case tests:
  - **1x1 matrices**: complete lifecycle (create/factor/solve/refine with both pivot strategies), extreme values (large and small), matvec
  - **Single non-zero**: off-diagonal singular detection, single diagonal element singular
  - **Diagonal extremes**: multi-scale diagonal (1e-10 to 1e10) solve, negative diagonal values
  - **Empty matrices**: copy empty, factor-singular detection, info accessors
  - **Negative indices**: remove, set, get all reject with SPARSE_ERR_BOUNDS
  - **Large/small values**: 2x2 solves with 1e150 and 1e-10 coefficients
  - **Modification**: deep copy independence through multiple mutations, insert/remove/re-insert same position
  - **Permutation**: perm/inv_perm consistency check after 6x6 factorization, factor with both strategies agree
  - **Free-list stress**: insert 2500 elements, remove all, re-insert — memory unchanged (slab reuse verified)
- Updated Makefile and CMakeLists.txt for new test target
- Reviewed `archive/` directory — already has comprehensive README from Day 1

### Test Results
- `test_sparse_matrix`: 31/31 passed, 166 assertions
- `test_sparse_lu`: 24/24 passed, 84 assertions
- `test_sparse_io`: 18/18 passed, 142 assertions
- `test_known_matrices`: 15/15 passed, 41 assertions
- `test_sparse_vector`: 24/24 passed, 52 assertions
- `test_edge_cases`: 20/20 passed, 114 assertions
- **Total: 132 unit tests, 599 assertions, all passing**

### Notes
- Discovered that backward_sub's `|u_ii| < DROP_TOL` check acts as an absolute threshold — values below 1e-14 are treated as singular regardless of the problem's scale. This is a known limitation documented in the algorithm description.
- The free-list reuse test confirms that after removing all 2500 entries and re-inserting, no additional slab allocations occur — memory_usage is identical.
- All 132 tests also pass under UBSan

## Day 9 — Integration Testing, CI Setup & Sprint Retrospective

### Completed
- Created `tests/test_integration.c` — 7 end-to-end integration tests:
  - **Load → factor → solve → residual → save**: loads tridiagonal_20.mtx, solves with known RHS, checks residual < 1e-12, saves and reloads
  - **Create → copy → factor → solve → refine → verify**: builds 10x10 pentadiagonal programmatically, preserves original via copy, solves, refines, checks solution accuracy
  - **Multiple RHS**: loads symmetric_4.mtx, factors once, solves with 3 different RHS vectors, checks all residuals
  - **Full round-trip**: creates 8x8 matrix, saves to MM, loads back, compares element-by-element, solves both and compares solutions
  - **All reference matrices**: loops over all 6 solvable reference matrices, factors and solves each with partial pivoting, checks relative residual
  - **Both pivots agree**: builds 15x15 matrix, solves with complete and partial pivoting, verifies solutions match
  - **Error recovery**: attempts singular factorization, handles error, then successfully factors and solves a different matrix
- Created `scripts/ci.sh` — CI script with options:
  - `--sanitize` to run UBSan build
  - `--bench` to run benchmarks (informational)
  - Reports pass/fail status
- Created `docs/planning/EPIC_1/SPRINT_1/RETROSPECTIVE.md`:
  - Definition of Done checklist (all 12 items complete)
  - Full review of all initial-review.md items with status (40+ items, all addressed or explicitly deferred)
  - 5 bugs found during sprint documented with fix descriptions
  - Final metrics table
  - 9 Sprint 2 candidate items
  - 5 lessons learned
- Updated Makefile and CMakeLists.txt for integration test target

### Test Results
- `test_sparse_matrix`: 31/31 passed, 166 assertions
- `test_sparse_lu`: 24/24 passed, 84 assertions
- `test_sparse_io`: 18/18 passed, 142 assertions
- `test_known_matrices`: 15/15 passed, 41 assertions
- `test_sparse_vector`: 24/24 passed, 52 assertions
- `test_edge_cases`: 20/20 passed, 114 assertions
- `test_integration`: 7/7 passed, 184 assertions
- **Total: 139 unit tests, 783 assertions, all passing**

### Notes
- Sprint 1 is complete. All items from the Definition of Done are satisfied.
- The only review item deferred is errno preservation (4.4) — low priority.
- Key Sprint 2 candidates: condition number estimation, Cholesky for SPD, fill-reducing reordering, relative drop tolerance.
