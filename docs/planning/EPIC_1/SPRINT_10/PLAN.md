# Sprint 10 Plan: CSR Acceleration, Block Operations & Packaging

**Sprint Duration:** 14 days
**Goal:** Convert the LU elimination inner loop to use a CSR working format for dramatic speedup on large matrices, add block operations for cache efficiency and multiple-RHS support, formalize CI-based line coverage reporting (verifying ≥95% from Sprint 9), and package the library for external use.

**Starting Point:** A sparse linear algebra library with ~120 public API functions, ~700 tests across ~35 test suites, 14 SuiteSparse reference matrices, LU/Cholesky/QR/SVD factorization (including zero-diagonal chase, condition number, partial SVD with vectors, sparse low-rank), CG/GMRES iterative solvers (matrix-based and matrix-free) with ILU(0)/ILUT preconditioning (left/right, pivoting), parallel SpMV, standalone examples, Doxygen API docs, fuzz and property-based tests, and ≥95% line coverage on library source. Profiling identified linked-list traversal as the fundamental LU bottleneck for large matrices.

**End State:** CSR working format for LU elimination (≥2x speedup on large matrices), block LU for cache-efficient dense subblock handling, block solvers for multiple RHS vectors (direct and iterative), CI-based line coverage reporting with automated ≥95% verification, and production-ready library packaging with `make install`, pkg-config, and CMake integration.

---

## Day 1: CSR Working Format — Data Structures & Conversion

**Theme:** Design CSR working format and implement conversion routines between linked-list and CSR representations

**Time estimate:** 10 hours

### Tasks
1. Design the CSR working format for LU elimination:
   - Define `LU_CSR` struct: `row_ptr[]`, `col_idx[]`, `values[]`, dimensions, nnz capacity
   - Plan the conversion pipeline: linked-list → CSR → elimination → CSR → linked-list
   - Decide on memory layout: separate L/U CSR or combined with diagonal marker
2. Implement `lu_csr_from_sparse()` in `src/sparse_lu.c`:
   - Convert the library's linked-list `SparseMatrix` into CSR format
   - Handle the column-linked-list traversal to build row-oriented arrays
   - Respect existing fill-reducing permutations (AMD/RCM row/col ordering)
3. Implement `lu_csr_to_sparse()`:
   - Convert CSR back to linked-list format after elimination
   - Reconstruct column chains and row chains from CSR arrays
   - Preserve permutation metadata
4. Write conversion round-trip tests:
   - Identity matrix: convert → convert back → verify exact match
   - Dense 5×5: verify all entries preserved
   - Sparse orsirr_1 (if available): verify round-trip preserves all entries and values
   - Empty matrix and 1×1 edge cases
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `LU_CSR` struct definition
- `lu_csr_from_sparse()` and `lu_csr_to_sparse()` conversion routines
- ≥4 round-trip conversion tests

### Completion Criteria
- Round-trip conversion preserves all matrix entries exactly
- No memory leaks under ASan
- `make format && make lint && make test` clean

---

## Day 2: CSR Working Format — Elimination Kernel

**Theme:** Implement the core LU elimination algorithm operating on CSR arrays

**Time estimate:** 12 hours

### Tasks
1. Implement `lu_csr_eliminate()` — the CSR-based LU factorization kernel:
   - Row-wise elimination with CSR arrays (no linked-list pointer chasing)
   - Pivot selection: partial pivoting with row swaps in CSR
   - Row operations: `row[k] -= (row[k][pivot_col] / pivot) * row[pivot]`
   - Fill-in management: dynamically grow CSR arrays when new nonzeros appear
   - Track L entries (below diagonal) and U entries (diagonal and above) in-place
2. Handle fill-in efficiently:
   - Pre-allocate CSR arrays with estimated fill (e.g., 2× initial nnz)
   - When capacity exceeded, realloc with growth factor
   - Keep a workspace row array for scatter-gather pattern
3. Implement the scatter-gather elimination pattern:
   - Scatter pivot row into dense workspace array
   - For each row below pivot: scatter, subtract scaled pivot row, gather nonzeros back
   - This avoids the linked-list traversal that dominates current LU cost
4. Write basic elimination tests:
   - 3×3 matrix with known LU factors: verify L*U = P*A
   - 5×5 matrix: compare CSR LU result with existing linked-list LU
   - Diagonal matrix: verify trivial factorization
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `lu_csr_eliminate()` kernel with scatter-gather pattern
- Fill-in management with dynamic CSR growth
- ≥3 elimination correctness tests

### Completion Criteria
- CSR LU matches existing linked-list LU results to machine precision
- Fill-in handled correctly (no lost entries)
- `make format && make lint && make test` clean

---

## Day 3: CSR Working Format — Integration, Solve & Benchmarking

**Theme:** Integrate CSR LU into the public API and benchmark speedup on large matrices

**Time estimate:** 10 hours

### Tasks
1. Integrate CSR path into `sparse_lu_factor()`:
   - Add a size threshold: use CSR path for matrices above a certain nnz or dimension
   - Convert to CSR → eliminate → convert back to linked-list for existing solve path
   - Alternatively, implement `lu_csr_solve()` for forward/backward substitution in CSR
2. Implement `lu_csr_solve()`:
   - Forward substitution with L in CSR (row-wise traversal)
   - Backward substitution with U in CSR (reverse row-wise)
   - Apply permutation vectors
3. Benchmark on SuiteSparse matrices:
   - orsirr_1 (1030×1030): target ≥2x speedup over linked-list LU
   - steam1 (240×240): measure speedup
   - nos4 (100×100): small matrix baseline
   - Record factor time, solve time, total time
4. Write integration tests:
   - Solve Ax=b with CSR path, compare solution with linked-list path
   - Verify residual ||Ax - b|| / ||b|| < tolerance
   - SuiteSparse matrix solve: verify against known solution
5. Run `make format && make lint && make test` — all clean

### Deliverables
- CSR path integrated into `sparse_lu_factor()` / `sparse_lu_solve()`
- Benchmark results on ≥3 SuiteSparse matrices
- ≥3 integration tests

### Completion Criteria
- ≥2x speedup on orsirr_1 LU factorization
- Solutions match linked-list path to machine precision
- `make format && make lint && make test` clean

---

## Day 4: Block LU — Dense Subblock Detection

**Theme:** Implement detection of dense subblocks within sparse matrices for block LU factorization

**Time estimate:** 10 hours

### Tasks
1. Design the block detection algorithm:
   - Scan the sparsity pattern to find dense submatrices (rectangular regions with high fill)
   - Use a threshold: a k×k subblock is "dense" if fill ratio > 80%
   - Consider supernodal detection: groups of consecutive columns with identical sparsity patterns
2. Implement `lu_detect_dense_blocks()` in `src/sparse_lu.c`:
   - Input: CSR sparsity pattern (row_ptr, col_idx)
   - Output: list of `DenseBlock` structs: {row_start, row_end, col_start, col_end}
   - Use the supernodal approach: find maximal column groups with identical nonzero row sets
   - Minimum block size: 4×4 (smaller blocks don't benefit from dense kernels)
3. Implement dense block extraction:
   - Extract a dense subblock from CSR into a column-major dense array
   - Insert a factored dense block back into CSR
4. Write detection tests:
   - Arrow matrix (dense first row/column + diagonal): detect the dense border
   - Block-diagonal matrix: detect each diagonal block
   - Fully sparse matrix (no dense blocks): empty detection result
   - Matrix with one large dense subblock: detect correct bounds
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `lu_detect_dense_blocks()` with supernodal column grouping
- Dense block extraction/insertion routines
- ≥4 detection tests

### Completion Criteria
- Correctly identifies dense subblocks in test matrices
- No false positives on fully sparse matrices
- `make format && make lint && make test` clean

---

## Day 5: Block LU — Dense Kernel & Integration

**Theme:** Implement dense LU kernels for detected subblocks and integrate with CSR elimination

**Time estimate:** 12 hours

### Tasks
1. Implement `lu_dense_factor()` — dense LU factorization for subblocks:
   - Column-major dense array, partial pivoting
   - BLAS-like implementation (dgetrf-style) without external BLAS dependency
   - Operate on the extracted dense subblock
2. Implement `lu_dense_solve()` — dense forward/backward substitution:
   - For a dense subblock's contribution during sparse elimination
   - Apply the dense L/U factors to update the Schur complement
3. Integrate block detection into CSR elimination pipeline:
   - During `lu_csr_eliminate()`, check if the current pivot region matches a detected dense block
   - If dense block detected: extract subblock → dense LU factor → update Schur complement
   - Otherwise: proceed with standard sparse scatter-gather elimination
4. Benchmark block LU:
   - Create test matrices with known dense substructure (e.g., block-diagonal + sparse coupling)
   - Compare block LU vs plain CSR LU timing
   - Measure cache miss reduction (if profiling tools available)
5. Write integration tests:
   - Block-diagonal matrix: block LU should produce same result as plain LU
   - Matrix with dense border: verify factorization correctness
   - Compare block LU solution with plain LU solution on SuiteSparse matrices
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `lu_dense_factor()` and `lu_dense_solve()` dense kernels
- Block detection integrated into CSR elimination
- ≥3 block LU integration tests

### Completion Criteria
- Block LU produces identical results to plain LU (to machine precision)
- Measurable speedup on matrices with dense substructure
- `make format && make lint && make test` clean

---

## Day 6: Block LU — Hardening & Edge Cases

**Theme:** Harden block LU with edge cases, near-singular blocks, and comprehensive testing

**Time estimate:** 8 hours

### Tasks
1. Handle edge cases in block detection and factoring:
   - Dense blocks with near-singular pivots: fall back to sparse elimination
   - Blocks at matrix boundary (last rows/columns): handle partial blocks
   - Overlapping block candidates: resolve by choosing the largest
   - Single-element "blocks": skip dense path (not worth overhead)
2. Test near-singular dense subblocks:
   - Dense block with one near-zero pivot: verify correct fallback or controlled error
   - Dense block with exact zero: verify permutation resolves it
3. Test on SuiteSparse matrices:
   - orsirr_1: verify block LU result matches plain LU
   - bcsstk04 (SPD stiffness matrix): should have exploitable block structure
   - steam1: verify on a small thermodynamics matrix
4. Memory and sanitizer validation:
   - Run block LU tests under ASan: no leaks, no out-of-bounds
   - Run under UBSan: no undefined behavior in dense kernels
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Edge case handling for boundary blocks, near-singular pivots
- SuiteSparse validation for block LU
- ASan/UBSan clean

### Completion Criteria
- Block LU handles all edge cases without crashes
- SuiteSparse results match plain LU to machine precision
- `make format && make lint && make test` clean, sanitizers clean

---

## Day 7: Block LU Solve — Multiple RHS

**Theme:** Implement block LU solve for multiple right-hand side vectors simultaneously

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_lu_solve_block()` in `src/sparse_lu.c`:
   - Signature: `sparse_err_t sparse_lu_solve_block(const SparseMatrix *LU, const double *B, idx_t nrhs, double *X)`
   - B is m×nrhs column-major, X is n×nrhs column-major
   - Forward substitution on all RHS vectors simultaneously
   - Backward substitution on all RHS vectors simultaneously
2. Optimize for cache efficiency:
   - Process multiple RHS vectors per row during forward/backward substitution
   - Use dense BLAS-like inner loops for the RHS block
   - Amortize sparse pattern traversal across all RHS vectors
3. Declare in public header `include/sparse_lu.h`
4. Write tests:
   - Single RHS: result matches existing `sparse_lu_solve()`
   - 5 RHS vectors: verify each solution independently
   - 10×10 matrix with 3 RHS: verify ||AX - B|| < tol for each column
   - Edge cases: nrhs=0, nrhs=1, nrhs > n
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_lu_solve_block()` in public API
- Cache-efficient multi-RHS forward/backward substitution
- ≥4 block solve tests

### Completion Criteria
- Multi-RHS solutions match single-RHS solutions to machine precision
- Measurable speedup vs solving nrhs times individually (for nrhs ≥ 4)
- `make format && make lint && make test` clean

---

## Day 8: Block CG Solver

**Theme:** Implement block conjugate gradient for multiple RHS vectors on SPD systems

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_cg_solve_block()` in `src/sparse_iterative.c`:
   - Signature: `sparse_err_t sparse_cg_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X, const sparse_iter_opts_t *opts)`
   - Block CG algorithm: simultaneous CG iteration for all RHS vectors
   - Shared SpMV: compute A*X_block in one pass (amortize matrix traversal)
   - Per-column convergence tracking: each RHS can converge independently
2. Implement block SpMV helper:
   - `sparse_matvec_block()`: multiply sparse matrix by dense matrix (m×nrhs)
   - Row-wise traversal, dot products against each RHS column
3. Declare in public header `include/sparse_iterative.h`
4. Write tests:
   - SPD 10×10 matrix with 3 RHS: verify each solution
   - Compare with single-RHS `sparse_cg_solve()` on each column
   - Preconditioned block CG: verify compatibility with existing preconditioners
   - Convergence test: verify iteration count similar to single-RHS CG
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_cg_solve_block()` in public API
- `sparse_matvec_block()` helper
- ≥4 block CG tests

### Completion Criteria
- Block CG solutions match single-RHS CG to convergence tolerance
- Shared SpMV reduces total matrix traversals
- `make format && make lint && make test` clean

---

## Day 9: Block GMRES Solver

**Theme:** Implement block GMRES for multiple RHS vectors on general systems

**Time estimate:** 10 hours

### Tasks
1. Implement `sparse_gmres_solve_block()` in `src/sparse_iterative.c`:
   - Signature: `sparse_err_t sparse_gmres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X, const sparse_iter_opts_t *opts)`
   - Block GMRES: shared Krylov subspace construction across RHS vectors
   - Block Arnoldi process: orthogonalize block vectors
   - Per-column convergence tracking with deflation of converged columns
2. Handle block-specific challenges:
   - Deflation: when one RHS converges, remove it from the active block
   - Restart: block GMRES(m) with configurable restart length
   - Preconditioning: apply existing preconditioners column-wise
3. Declare in public header `include/sparse_iterative.h`
4. Write tests:
   - General 10×10 matrix with 3 RHS: verify each solution
   - Compare with single-RHS `sparse_gmres_solve()` on each column
   - Preconditioned block GMRES with ILU(0)
   - Non-symmetric matrix: verify block GMRES handles asymmetry
   - Convergence stalling: verify restart mechanism works
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_gmres_solve_block()` in public API
- Block Arnoldi with deflation
- ≥5 block GMRES tests

### Completion Criteria
- Block GMRES solutions match single-RHS GMRES to convergence tolerance
- Deflation works correctly when columns converge at different rates
- `make format && make lint && make test` clean

---

## Day 10: Line Coverage Measurement & CI Integration

**Theme:** Set up CI-based coverage reporting and verify ≥95% line coverage from Sprint 9

**Time estimate:** 8 hours

### Tasks
1. Add `make coverage` Makefile target:
   - Compile library and tests with `--coverage` (GCC) or equivalent flags
   - Run the full test suite to generate `.gcda` coverage data
   - Use `lcov` to collect coverage data into a `.info` file
   - Use `genhtml` to produce an HTML coverage report in `coverage/`
   - Print summary line coverage percentage to stdout
2. Add coverage to GitHub Actions CI:
   - New CI job or step that runs `make coverage`
   - Upload HTML report as artifact
   - Fail the build if line coverage on `src/*.c` drops below 95%
   - Use `lcov --fail-under-lines 95` or equivalent threshold check
3. Run coverage locally and analyze results:
   - Identify any uncovered lines in `src/*.c`
   - If coverage is below 95%, add targeted tests to cover gaps
   - Focus on error paths, edge cases, and rarely-exercised branches
4. Add `.gitignore` entries for coverage artifacts (`*.gcda`, `*.gcno`, `coverage/`)
5. Run `make format && make lint && make test && make coverage` — all clean

### Deliverables
- `make coverage` target producing HTML report
- CI integration with 95% threshold enforcement
- Coverage report showing ≥95% on `src/*.c`

### Completion Criteria
- `make coverage` produces accurate HTML report
- CI enforces ≥95% line coverage threshold
- Any coverage gaps from Sprint 9 identified and filled
- `make format && make lint && make test` clean

---

## Day 11: Packaging — Makefile Install & pkg-config

**Theme:** Add `make install` target and pkg-config support for library distribution

**Time estimate:** 10 hours

### Tasks
1. Implement `make install` target:
   - Configurable `PREFIX` (default: `/usr/local`)
   - Install shared library (`libsparse.so` / `libsparse.dylib`) to `$(PREFIX)/lib/`
   - Install static library (`libsparse.a`) to `$(PREFIX)/lib/`
   - Install public headers from `include/` to `$(PREFIX)/include/sparse/`
   - Set proper file permissions and run `ldconfig` on Linux
2. Implement `make uninstall` target:
   - Remove installed files
3. Create pkg-config `.pc` file (`sparse.pc`):
   - Template with `@PREFIX@` substitution during install
   - Correct `-I` include path and `-L` library path
   - `-lsparse -lm` link flags
   - Version string from a central `VERSION` file or macro
4. Add library versioning:
   - Define `SPARSE_VERSION_MAJOR`, `SPARSE_VERSION_MINOR`, `SPARSE_VERSION_PATCH` in a header
   - Shared library soname versioning (e.g., `libsparse.so.1.0.0`)
5. Write install/uninstall tests:
   - Install to a temp directory (`PREFIX=/tmp/sparse_test_install`)
   - Verify all expected files exist with correct permissions
   - Compile an example program against the installed library using pkg-config
   - Uninstall and verify files removed
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `make install` / `make uninstall` targets
- `sparse.pc` pkg-config file
- Library versioning headers
- ≥3 install validation tests (scripted)

### Completion Criteria
- `make install PREFIX=/tmp/test` installs all files correctly
- `pkg-config --cflags --libs sparse` returns correct flags
- Example program compiles and links against installed library
- `make format && make lint && make test` clean

---

## Day 12: Packaging — CMake Integration & Cross-Platform

**Theme:** Add CMake find_package support and write cross-platform installation instructions

**Time estimate:** 8 hours

### Tasks
1. Create CMake config files for `find_package(Sparse)`:
   - `SparseConfig.cmake`: define imported target `Sparse::sparse`
   - `SparseConfigVersion.cmake`: version compatibility checking
   - Install these to `$(PREFIX)/lib/cmake/Sparse/`
2. Write a CMake example project (`examples/cmake_example/`):
   - `CMakeLists.txt` that uses `find_package(Sparse REQUIRED)`
   - Links against `Sparse::sparse`
   - Build and verify the example links and runs correctly
3. Test CMake integration:
   - Install library → build CMake example → verify it works
   - Test with both shared and static library
4. Write installation instructions (`INSTALL.md`):
   - Linux (Ubuntu/Debian, Fedora): build from source, install, link
   - macOS: build with Clang, install, link (Homebrew-friendly)
   - Windows (MSVC): build with MSVC, static linking instructions
   - Common issues and troubleshooting
5. Test build on available platforms:
   - Verify `make && make test && make install` works on current platform
   - Verify CI build still passes
6. Run `make format && make lint && make test` — all clean

### Deliverables
- CMake `find_package` config files
- CMake example project
- `INSTALL.md` with cross-platform instructions

### Completion Criteria
- `find_package(Sparse)` works in CMake projects
- CMake example builds and runs against installed library
- Installation instructions cover Linux, macOS, Windows
- `make format && make lint && make test` clean

---

## Day 13: Integration Testing & Final Validation

**Theme:** Cross-feature integration testing, benchmark validation, and final hardening

**Time estimate:** 10 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — ASan/UBSan clean on all new code
   - `make bench` — benchmarks run without crashes
   - `make format && make lint` — all clean
   - `make coverage` — verify ≥95% line coverage maintained
2. CSR LU integration tests:
   - Solve linear systems on all available SuiteSparse matrices via CSR path
   - Verify residuals match linked-list path to machine precision
   - Benchmark CSR vs linked-list on orsirr_1: confirm ≥2x speedup
3. Block solver integration tests:
   - Block LU solve + Block CG + Block GMRES on same system with 5 RHS
   - Verify all three solvers produce consistent solutions
   - Test block solvers with preconditioning
4. Packaging integration tests:
   - Fresh install to temp directory
   - Build all examples against installed library (both Makefile and CMake)
   - Verify pkg-config and CMake find_package both work
5. Backward compatibility:
   - All Sprint 9 and earlier APIs work unchanged
   - No breaking API changes
   - Existing test suite passes without modification

### Deliverables
- Full regression pass under all sanitizers
- Cross-feature integration tests for CSR LU, block solvers, packaging
- Benchmark validation confirming speedup targets

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 findings
- CSR LU achieves ≥2x speedup on orsirr_1
- All packaging paths verified
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final documentation updates, sprint review, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Update documentation:
   - Update `README.md` with new API functions (block solvers, CSR LU) and test counts
   - Update Doxygen comments for all new public functions
   - Run `make docs` to regenerate API reference
   - Verify tutorial and examples reflect final API
2. Final metrics collection:
   - Total test count
   - Total public API function count
   - Line coverage percentage
   - Benchmark speedup summary (CSR LU vs linked-list)
3. Write `docs/planning/EPIC_1/SPRINT_10/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Performance improvements achieved
   - Final metrics (test count, API functions, coverage, speedup)
   - Items deferred (if any)
4. Tag release:
   - Create git tag `v1.0.0` (or appropriate version) if all items complete
   - Update VERSION file/macro

### Deliverables
- Updated README and API documentation
- Sprint retrospective document
- Release tag (if appropriate)

### Completion Criteria
- All documentation current and accurate
- Retrospective written with honest assessment
- `make format && make lint && make test && make docs` clean

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | CSR format — data structures & conversion | 10 | `LU_CSR` struct, conversion routines, ≥4 tests |
| 2 | CSR format — elimination kernel | 12 | `lu_csr_eliminate()`, scatter-gather, ≥3 tests |
| 3 | CSR format — integration & benchmarking | 10 | CSR path in public API, ≥2x speedup, ≥3 tests |
| 4 | Block LU — dense subblock detection | 10 | `lu_detect_dense_blocks()`, ≥4 tests |
| 5 | Block LU — dense kernel & integration | 12 | Dense LU kernels, block detection integration, ≥3 tests |
| 6 | Block LU — hardening & edge cases | 8 | Edge cases, SuiteSparse validation, sanitizer clean |
| 7 | Block LU solve — multiple RHS | 10 | `sparse_lu_solve_block()`, ≥4 tests |
| 8 | Block CG solver | 10 | `sparse_cg_solve_block()`, ≥4 tests |
| 9 | Block GMRES solver | 10 | `sparse_gmres_solve_block()`, ≥5 tests |
| 10 | Line coverage measurement | 8 | `make coverage`, CI integration, ≥95% verified |
| 11 | Packaging — install & pkg-config | 10 | `make install`, `sparse.pc`, versioning |
| 12 | Packaging — CMake & cross-platform | 8 | CMake find_package, `INSTALL.md` |
| 13 | Integration testing & validation | 10 | Full regression, cross-feature tests, benchmarks |
| 14 | Sprint review & retrospective | 4 | Docs update, retrospective, release tag |

**Total estimate:** 132 hours (avg ~9.4 hrs/day, max 12 hrs/day)
