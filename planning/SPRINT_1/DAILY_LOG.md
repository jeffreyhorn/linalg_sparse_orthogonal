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
