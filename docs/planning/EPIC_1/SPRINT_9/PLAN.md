# Sprint 9 Plan: SVD Hardening, Performance & Documentation

**Sprint Duration:** 14 days
**Goal:** Complete SVD feature set by fixing rank-deficient convergence (zero-diagonal chase), adding SVD-based condition number estimation, recovering singular vectors from partial SVD, and producing sparse low-rank output. Then optimize performance across the library, write examples and documentation, and harden the test suite.

**Starting Point:** A sparse linear algebra library with ~100 public API functions, ~600 tests across ~30 test suites, 14 SuiteSparse reference matrices, LU/Cholesky/QR/SVD factorization, CG/GMRES iterative solvers (matrix-based and matrix-free) with ILU(0)/ILUT preconditioning (left/right, pivoting), parallel SpMV, full SVD (`sparse_svd_compute()`), partial SVD (`sparse_svd_partial()` — singular values only), pseudoinverse (`sparse_pinv()`), dense low-rank approximation (`sparse_svd_lowrank()`), SVD-based rank estimation, CI/CD with GitHub Actions, and code quality tooling.

**End State:** Zero-diagonal chase for rank-deficient SVD, `sparse_cond()`, partial SVD with singular vectors, sparse low-rank output, performance-optimized SVD and factorizations, standalone examples, Doxygen API docs, and hardened test suite with ≥95% coverage.

---

## Day 1: Zero-Diagonal Chase — Algorithm Design & Core Implementation

**Theme:** Implement the zero-diagonal chase algorithm (Golub & Van Loan §8.6.2) for rank-deficient bidiagonals

**Time estimate:** 12 hours

### Tasks
1. Study the zero-diagonal chase algorithm from Golub & Van Loan §8.6.2:
   - When a diagonal entry B(i,i) ≈ 0, the standard implicit QR step fails
   - The chase applies Givens rotations to zero the superdiagonal entry B(i-1,i)
   - Rotations propagate up (or down) the bidiagonal, creating and chasing a bulge
2. Implement `bidiag_zero_diag_chase()` in `src/sparse_svd.c`:
   - Detect near-zero diagonal entries: |B(i,i)| < tol * ||B||
   - Apply Givens rotations to eliminate the associated superdiagonal entry
   - Chase the resulting bulge to the boundary of the bidiagonal
   - Accumulate rotations into U (left rotations) and V (right rotations)
3. Integrate into `bidiag_svd_iterate()`:
   - Before each QR step, check for near-zero diagonal entries
   - If found, apply zero-diagonal chase instead of QR step
   - After chase, the row/column can be deflated (split bidiagonal)
4. Write initial tests:
   - Rank-1 matrix (single nonzero singular value): verify convergence
   - Rank-2 matrix in a 5×5: verify two nonzero singular values found
   - Matrix with exact zero diagonal entry: verify immediate deflation
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `bidiag_zero_diag_chase()` function
- Integration with `bidiag_svd_iterate()`
- ≥3 rank-deficient SVD tests

### Completion Criteria
- Rank-1 SVD test passes (previously disabled)
- Rank-deficient singular values correct to 1e-10
- `make format && make lint && make test` clean

---

## Day 2: Zero-Diagonal Chase — Testing & Edge Cases

**Theme:** Validate zero-diagonal chase on diverse rank-deficient matrices

**Time estimate:** 10 hours

### Tasks
1. Re-enable the previously disabled rank-1 SVD test with strict assertion
2. Test on rank-deficient matrices of various sizes:
   - 3×3 rank-1: verify exactly one nonzero singular value
   - 5×5 rank-2: verify exactly two nonzero singular values
   - 10×10 rank-5: verify five nonzero + five near-zero singular values
3. Test on matrices with multiple zero diagonals:
   - Bidiagonal with alternating zero/nonzero diagonals
   - Bidiagonal with consecutive zero diagonals
4. Test on near-singular matrices:
   - Diagonal entries approaching machine epsilon
   - Verify graceful degradation vs exact zero
5. SuiteSparse validation:
   - Create a rank-deficient matrix from SuiteSparse by zeroing columns
   - Verify SVD produces correct rank
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Rank-1 SVD test re-enabled and passing
- ≥6 additional rank-deficient SVD tests
- Near-singular edge cases covered

### Completion Criteria
- All rank-deficient tests pass with correct singular value counts
- Zero singular values are < 1e-10 * sigma_max
- `make format && make lint && make test` clean

---

## Day 3: Condition Number Estimation via SVD

**Theme:** Implement `sparse_cond()` for condition number estimation

**Time estimate:** 8 hours

### Tasks
1. Declare `sparse_cond()` in `include/sparse_svd.h`:
   - `double sparse_cond(const SparseMatrix *A, sparse_err_t *err);`
   - Returns cond(A) = sigma_max / sigma_min
   - Sets `*err` to SPARSE_OK on success, error code on failure
2. Implement in `src/sparse_svd.c`:
   - For small matrices (min(m,n) ≤ 100): use full SVD
   - For larger matrices: use partial SVD for sigma_max (k=1) and inverse iteration or full SVD for sigma_min
   - Return `INFINITY` when sigma_min is below tolerance (singular matrix)
   - Handle rectangular matrices: use min(m,n) singular values
3. Write tests:
   - Identity matrix: cond = 1.0
   - Diagonal matrix with known condition number
   - Well-conditioned SPD matrix (nos4): verify reasonable condition number
   - Ill-conditioned matrix: verify large condition number
   - Singular matrix: verify returns INFINITY
   - 1×1 matrix: cond = 1.0 for nonzero, INFINITY for zero
   - Rectangular matrix: condition number of min(m,n) singular values
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_cond()` in public API
- ≥7 condition number tests

### Completion Criteria
- Identity matrix returns cond = 1.0 exactly
- Singular matrix returns INFINITY
- Known condition numbers match to 1e-10
- `make format && make lint && make test` clean

---

## Day 4: SVD Singular Vectors for Partial SVD — Core Implementation

**Theme:** Extend `sparse_svd_partial()` to recover approximate singular vectors from Lanczos basis

**Time estimate:** 12 hours

### Tasks
1. Modify Lanczos bidiagonalization in `sparse_svd_partial()`:
   - When `opts->compute_uv` is set, store the Lanczos P vectors (m×k) and Q vectors (n×(k+1))
   - P columns are left Lanczos vectors, Q columns are right Lanczos vectors
   - Allocate storage for P and Q matrices (column-major dense arrays)
2. After bidiagonal SVD converges on the k×k bidiagonal:
   - Obtain left/right singular vectors of the small bidiagonal: U_small (k×k), V_small (k×k)
   - Recover approximate singular vectors: U ≈ P * U_small, V ≈ Q(:,1:k) * V_small
3. Store results in `sparse_svd_t`:
   - `U` is m×k dense array (column-major)
   - `Vt` is k×n dense array (row-major, transposed V)
4. Update `sparse_svd_free()` to handle partial SVD allocations
5. Write initial tests:
   - 10×10 matrix, k=3: verify U^T*U ≈ I_3 and V^T*V ≈ I_3
   - Verify A*v_i ≈ sigma_i * u_i for each singular triplet
   - Compare partial U/V columns with full SVD U/V columns (up to sign)
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_svd_partial()` with `compute_uv` support
- Lanczos P/Q vector storage and singular vector recovery
- ≥3 partial SVD vector tests

### Completion Criteria
- Partial singular vectors orthonormal to 1e-8
- A*v_i ≈ sigma_i * u_i to 1e-8
- `make format && make lint && make test` clean

---

## Day 5: SVD Singular Vectors for Partial SVD — Validation

**Theme:** Validate partial SVD vectors against full SVD on real matrices

**Time estimate:** 10 hours

### Tasks
1. Test on SuiteSparse matrices:
   - nos4 (100×100), k=5: compare top-5 singular vectors with full SVD
   - bcsstk04 (132×132), k=5: verify on stiffness matrix with clustered values
   - west0067 (67×67), k=3: unsymmetric case
2. Reconstruction tests:
   - Verify ||A - sum_{i=1}^{k} sigma_i * u_i * v_i^T|| ≈ sigma_{k+1} (from full SVD)
   - Best rank-k approximation property
3. Edge cases:
   - k=1: single largest singular triplet
   - k = min(m,n): should approximate full SVD
   - Rectangular wide matrix (m < n), k=3
4. Verify `compute_uv=0` still works (singular values only, no vector storage)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- ≥6 partial SVD vector validation tests
- SuiteSparse validation for partial vectors
- Reconstruction error tests

### Completion Criteria
- Partial vectors match full SVD vectors (up to sign) within 1e-6
- Reconstruction error ≈ sigma_{k+1}
- `make format && make lint && make test` clean

---

## Day 6: Sparse Low-Rank Output

**Theme:** Implement `sparse_svd_lowrank_sparse()` for memory-efficient sparse low-rank approximation

**Time estimate:** 8 hours

### Tasks
1. Declare `sparse_svd_lowrank_sparse()` in `include/sparse_svd.h`:
   - `sparse_err_t sparse_svd_lowrank_sparse(const SparseMatrix *A, idx_t k, double drop_tol, SparseMatrix **result);`
   - Computes best rank-k approximation and returns as sparse matrix
   - Entries with |value| < drop_tol are dropped (thresholded)
2. Implement in `src/sparse_svd.c`:
   - Compute SVD (full or partial depending on k vs min(m,n))
   - Form rank-k approximation: A_k = sum_{i=1}^{k} sigma_i * u_i * v_i^T
   - Insert entries into sparse matrix, skipping those below drop_tol
   - If drop_tol ≤ 0, use default: eps * sigma_1
3. Write tests:
   - Diagonal matrix rank-k: exact result, all entries preserved
   - Dense random 10×10, k=2, drop_tol=0.1: verify sparsity reduction
   - Compare with dense `sparse_svd_lowrank()`: ||sparse_result - dense_result||_F < drop_tol * sqrt(m*n)
   - Zero drop_tol: sparse result matches dense result exactly
   - k=1 on rank-1 matrix: result equals original (within tolerance)
   - Rectangular matrix: verify dimensions correct
4. Run `make format && make lint && make test` — all clean

### Deliverables
- `sparse_svd_lowrank_sparse()` in public API
- ≥6 sparse low-rank tests

### Completion Criteria
- Sparse result approximates dense low-rank within drop_tol bounds
- Sparse output has fewer nonzeros than original for non-trivial drop_tol
- `make format && make lint && make test` clean

---

## Day 7: SVD Performance Profiling

**Theme:** Profile SVD on SuiteSparse matrices and identify optimization targets

**Time estimate:** 10 hours

### Tasks
1. Create a profiling harness (`tests/bench_svd.c` or extend existing benchmarks):
   - Time full SVD on nos4, west0067, bcsstk04, steam1
   - Time partial SVD (k=5) on same matrices
   - Break down: bidiagonalization time vs QR iteration time vs U/V formation time
2. Profile with Instruments or `gprof`:
   - Identify hot functions in SVD pipeline
   - Measure cache miss rates in bidiagonalization inner loop
   - Profile memory allocation patterns
3. Identify optimization targets:
   - Dense bidiagonalization: is Householder application cache-friendly?
   - Sparse-to-dense conversion overhead
   - Memory allocation in Lanczos (P, Q vector storage)
   - QR iteration: Givens rotation application patterns
4. Document findings in `docs/planning/EPIC_1/SPRINT_9/PROFILE_RESULTS.md`:
   - Timing breakdown per matrix
   - Top-5 hot functions
   - Optimization opportunities ranked by expected impact
5. Run `make format && make lint && make test` — all clean

### Deliverables
- SVD profiling harness
- Profile data on ≥4 SuiteSparse matrices
- `PROFILE_RESULTS.md` with analysis

### Completion Criteria
- Timing breakdown for each SVD phase documented
- Top optimization targets identified and ranked
- `make format && make lint && make test` clean

---

## Day 8: SVD Performance Optimization

**Theme:** Implement the highest-impact SVD optimizations identified in profiling

**Time estimate:** 12 hours

### Tasks
1. Cache-blocked Householder application (if profiling shows it's a bottleneck):
   - Apply Householder reflections in blocks of b columns (b = 32 or 64)
   - Form the compact WY representation: Q = I - V*T*V^T
   - Apply block reflector via Level-3 BLAS-like operations
   - Reduces memory traffic vs column-by-column application
2. Optimize Lanczos bidiagonalization:
   - Reduce redundant sparse matvec operations
   - Optimize reorthogonalization: skip when loss of orthogonality is small
   - Selective reorthogonalization based on estimated orthogonality level
3. Optimize QR iteration inner loop:
   - Minimize memory writes in Givens rotation application
   - Unroll small loops for 2×2 rotations
4. Benchmark after optimizations:
   - Re-run profiling harness on same matrices
   - Compare with Day 7 baseline
   - Target: ≥2x speedup on bcsstk04 SVD
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Cache-blocked Householder (if applicable)
- Optimized Lanczos reorthogonalization
- Benchmark comparison with baseline

### Completion Criteria
- Measurable speedup on bcsstk04 SVD (target ≥2x)
- All existing tests still pass (no accuracy regression)
- `make format && make lint && make test` clean

---

## Day 9: General Performance Profiling & Optimization

**Theme:** Profile and optimize factorization performance across the library

**Time estimate:** 12 hours

### Tasks
1. Profile all factorizations on large SuiteSparse matrices:
   - LU on orsirr_1, steam1
   - Cholesky on bcsstk04, nos4
   - QR on west0067
   - CG/GMRES convergence on nos4, orsirr_1
2. Identify hot paths:
   - Pool allocator: allocation/free frequency and cost
   - Node traversal: linked-list cache behavior
   - Pivot search: column scan patterns
   - Sparse matvec: row/column traversal efficiency
3. Implement optimizations:
   - Pool allocator: batch allocation, free-list optimization
   - Node traversal: prefetching hints for linked-list walks
   - Pivot search: cache pivot candidates
   - Dense inner loops: SIMD-friendly layout where applicable
4. Benchmark after optimizations:
   - Compare with baseline on same matrices
   - Target: ≥1.5x speedup on orsirr_1 LU factorization
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Factorization profiling data
- Implemented optimizations for hot paths
- Benchmark comparison showing speedup

### Completion Criteria
- Measurable speedup on orsirr_1 factorization (target ≥1.5x)
- All existing tests still pass
- `make format && make lint && make test` clean

---

## Day 10: Comprehensive Examples & Tutorials — Part 1

**Theme:** Write standalone example programs for basic use cases

**Time estimate:** 10 hours

### Tasks
1. Create `examples/` directory with standalone C programs:
   - `example_basic_solve.c`: Load a matrix, solve Ax=b with LU, print result
   - `example_least_squares.c`: Overdetermined system via QR, print residual
   - `example_svd_lowrank.c`: SVD low-rank approximation, show compression ratio
   - `example_iterative.c`: Preconditioned GMRES on a large system
2. Each example should:
   - Be self-contained (single file, include only public headers)
   - Include explanatory comments
   - Compile with a simple `cc -o example example.c -lsparse -lm`
   - Produce readable output showing the computation
3. Add example build targets to Makefile:
   - `make examples` builds all examples
   - Each example is an independent target
4. Write a `examples/README.md` with usage instructions
5. Run `make format && make lint && make test && make examples` — all clean

### Deliverables
- 4 standalone example programs
- Example build targets in Makefile
- `examples/README.md`

### Completion Criteria
- All examples compile and run successfully
- Output is clear and educational
- `make format && make lint && make test` clean

---

## Day 11: Comprehensive Examples & Tutorials — Part 2

**Theme:** Write tutorial document and additional advanced examples

**Time estimate:** 10 hours

### Tasks
1. Write additional example programs:
   - `example_condition.c`: Estimate condition number, demonstrate ill-conditioning
   - `example_matrix_free.c`: Matrix-free GMRES with a custom operator callback
2. Write `docs/tutorial.md`:
   - Getting Started: building the library, linking
   - Section 1: Creating and manipulating sparse matrices
   - Section 2: Direct solvers (LU, Cholesky, QR)
   - Section 3: Iterative solvers (CG, GMRES, preconditioning)
   - Section 4: SVD and applications (rank, pseudoinverse, low-rank)
   - Section 5: Matrix-free interface
   - Each section with code snippets and expected output
3. Verify all code snippets in the tutorial compile and run
4. Run `make format && make lint && make test` — all clean

### Deliverables
- 2 additional example programs
- `docs/tutorial.md` covering all major library features

### Completion Criteria
- Tutorial covers all major API areas
- All code snippets in tutorial are verified working
- `make format && make lint && make test` clean

---

## Day 12: API Documentation Generator

**Theme:** Set up Doxygen for automated API reference generation

**Time estimate:** 8 hours

### Tasks
1. Create `Doxyfile` configuration:
   - Input: `include/` directory (all public headers)
   - Output: `docs/api/` (HTML)
   - Extract all: functions, typedefs, structs, enums, macros
   - Enable call graphs and source browser
2. Audit and complete doc comments in public headers:
   - `include/sparse_types.h`: all types and error codes
   - `include/sparse_matrix.h`: matrix creation, manipulation
   - `include/sparse_lu.h`, `sparse_cholesky.h`, `sparse_qr.h`: factorizations
   - `include/sparse_iterative.h`: CG, GMRES, matrix-free
   - `include/sparse_svd.h`: SVD, condition number, pseudoinverse
   - `include/sparse_ilu.h`: ILU(0), ILUT
   - Ensure every public function has: brief description, param docs, return docs
3. Add Makefile target:
   - `make docs` runs Doxygen and generates HTML
4. Verify generated documentation:
   - All public functions appear
   - No undocumented warnings
   - Cross-references work
5. Run `make format && make lint && make test && make docs` — all clean

### Deliverables
- `Doxyfile` configuration
- Complete doc comments on all public API functions
- `make docs` target generating HTML reference
- Generated HTML in `docs/api/`

### Completion Criteria
- Doxygen produces clean output with no warnings
- All public functions documented
- `make format && make lint && make test` clean

---

## Day 13: Final Test Hardening

**Theme:** Add fuzz testing, property-based tests, and achieve ≥95% line coverage

**Time estimate:** 8 hours

### Tasks
1. Add fuzz testing for Matrix Market parser:
   - Create `tests/fuzz_mm_parser.c` with random/corrupted inputs
   - Exercise `sparse_load_mm()` with:
     - Truncated files, missing headers, wrong dimensions
     - Negative indices, out-of-range values, NaN/Inf
     - Extremely large dimensions (overflow checks)
     - Empty file, binary garbage, UTF-8 sequences
   - Verify no crashes, memory leaks, or undefined behavior
2. Add property-based tests:
   - Random SPD matrices: Cholesky factor → solve → verify residual < tol
   - Random general matrices: LU factor → solve → verify residual < tol
   - Random overdetermined systems: QR solve → verify residual minimality
   - Random matrices: SVD → verify A ≈ U*Σ*V^T
3. Check line coverage:
   - Run `make coverage` (add target if needed using `--coverage` flags)
   - Identify uncovered lines in library source
   - Add targeted tests for uncovered branches
   - Target: ≥95% line coverage on `src/*.c`
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Fuzz tests for Matrix Market parser
- Property-based tests for all factorizations
- Coverage report showing ≥95% line coverage

### Completion Criteria
- Fuzz tests exercise ≥20 malformed inputs without crashes
- Property-based tests pass on ≥10 random matrices each
- Line coverage ≥95% on library source
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final validation, cross-feature integration, cleanup, and retrospective

**Time estimate:** 8 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan/ASan clean
   - `make bench` — benchmarks run, no crashes
   - `make format && make lint` — all clean
   - `make docs` — Doxygen clean
2. Cross-feature integration tests:
   - Zero-diagonal chase on rank-1 matrix → correct SVD
   - `sparse_cond()` on ill-conditioned matrix matches manual sigma_max/sigma_min
   - Partial SVD with vectors → reconstruction matches full SVD
   - Sparse low-rank output → approximation error within bounds
   - Performance benchmarks show improvement over Sprint 8 baseline
3. Verify backward compatibility:
   - All Sprint 8 and earlier APIs work unchanged
   - No breaking API changes
4. Update documentation:
   - Update `README.md` with new API functions and test counts
   - Verify tutorial and examples reflect final API
5. Write `docs/planning/EPIC_1/SPRINT_9/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Performance improvements achieved
   - Final metrics (test count, API functions, coverage)
   - Items deferred to Sprint 10

### Deliverables
- All tests pass under all sanitizers
- Cross-feature integration tests
- Updated documentation
- Sprint retrospective document

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 findings
- `make bench` completes without error
- `make format && make lint` clean
- `make docs` clean
- Retrospective written with honest assessment

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | Zero-diagonal chase — core | 12 | `bidiag_zero_diag_chase()`, rank-deficient SVD |
| 2 | Zero-diagonal chase — testing | 10 | Rank-1 test re-enabled, ≥6 edge-case tests |
| 3 | Condition number estimation | 8 | `sparse_cond()`, ≥7 tests |
| 4 | Partial SVD vectors — core | 12 | Lanczos P/Q storage, vector recovery, ≥3 tests |
| 5 | Partial SVD vectors — validation | 10 | SuiteSparse validation, reconstruction tests |
| 6 | Sparse low-rank output | 8 | `sparse_svd_lowrank_sparse()`, ≥6 tests |
| 7 | SVD performance profiling | 10 | Profiling harness, `PROFILE_RESULTS.md` |
| 8 | SVD performance optimization | 12 | Cache-blocked Householder, optimized Lanczos |
| 9 | General performance optimization | 12 | Factorization hot-path optimization, benchmarks |
| 10 | Examples & tutorials — part 1 | 10 | 4 example programs, Makefile targets |
| 11 | Examples & tutorials — part 2 | 10 | 2 more examples, `docs/tutorial.md` |
| 12 | API documentation generator | 8 | Doxyfile, doc comments, `make docs` |
| 13 | Final test hardening | 8 | Fuzz tests, property-based tests, ≥95% coverage |
| 14 | Sprint review & retrospective | 8 | Integration tests, docs update, retrospective |

**Total estimate:** 138 hours (avg ~9.9 hrs/day, max 12 hrs/day)
