# Project Plan: linalg_sparse_orthogonal — Sprints 2–9

Items deferred from Sprint 1 (see `SPRINT_1/RETROSPECTIVE.md`), organized into sprints with logical dependency ordering.

---

## Sprint 2: Hardening & Arithmetic Extensions (COMPLETE)

**Duration:** 14 days (~67 hours actual)

**Goal:** Shore up robustness gaps from Sprint 1, add fundamental matrix arithmetic, and establish a larger test corpus for validating future features.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | errno preservation | Map system `errno` to `sparse_err_t` codes in all I/O paths (`fopen`, `fread`, `fscanf` failures). Add `SPARSE_ERR_IO` with optional errno snapshot. | 6 hrs |
| 2 | Relative drop tolerance | Replace absolute `DROP_TOL` in backward substitution's singular-check with a tolerance relative to the matrix norm (e.g., `||A||_inf * eps`). Compute and cache the norm during factorization. | 10 hrs |
| 3 | ASan validation | Resolve macOS sandbox ASan hang. Test on Linux or native macOS. Add ASan build target to Makefile/CMake. Run full test suite under ASan and fix any findings. | 10 hrs |
| 4 | Sparse matrix addition/scaling | Implement `sparse_add(A, B, alpha, beta)` for `C = alpha*A + beta*B`. Support in-place variant. Add scalar scaling `sparse_scale(A, alpha)`. Unit tests with identity, zero, and rectangular matrices. | 16 hrs |
| 5 | Larger reference matrices | Download real SuiteSparse matrices (e.g., west0479, bcsstk14, fidap007). Write download/conversion script. Add benchmark runs against these matrices. Validate existing solver on matrices with ≥1000 nonzeros. | 12 hrs |

### Deliverables

- All I/O errors report meaningful `sparse_err_t` codes with errno context
- Backward substitution uses relative tolerance; ill-conditioned matrices no longer false-trigger singularity
- Full test suite passes under both ASan and UBSan
- `sparse_add()` and `sparse_scale()` in public API with tests
- ≥5 real-world SuiteSparse matrices in test corpus with benchmark results

**Total estimate:** ~54 hours

---

## Sprint 3: Numerical Robustness & Fill-Reducing Reordering (COMPLETE)

**Duration:** 14 days (~81 hours actual)

**Goal:** Add condition number estimation for diagnostics and implement fill-reducing reordering (AMD/RCM) to improve factorization performance on larger matrices.

### Prerequisites from Sprint 2

- Relative drop tolerance (needed for robust condition number estimation)
- Sparse matrix addition/scaling (needed for norm computations in condition estimation)
- Larger reference matrices (needed for validating reordering effectiveness)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Condition number estimation | Estimate `cond(A)` from LU factors using Hager's algorithm (1-norm estimator). Return `sparse_lu_condest()` alongside factorization. Warn user when condition number exceeds threshold. | 24 hrs |
| 2 | Fill-reducing reordering (AMD/RCM) | Implement Approximate Minimum Degree (AMD) or Reverse Cuthill-McKee (RCM) ordering. Integrate as optional pre-processing step in `sparse_lu_factor()` via options struct. Benchmark fill-in reduction on SuiteSparse matrices. | 40 hrs |

### Deliverables

- `sparse_lu_condest()` returns 1-norm condition estimate from existing LU factors
- Solver warns on ill-conditioned systems (condition number > user-configurable threshold)
- `sparse_reorder_amd()` and/or `sparse_reorder_rcm()` in public API
- Factorization options struct supports `SPARSE_REORDER_NONE`, `SPARSE_REORDER_AMD`, `SPARSE_REORDER_RCM`
- Benchmark data showing fill-in reduction on ≥3 SuiteSparse matrices
- All existing tests continue to pass; new tests for reordering correctness

**Total estimate:** ~64 hours

---

## Sprint 4: Cholesky Factorization & Thread Safety

**Duration:** 14 days (~120 hours)

**Goal:** Add Cholesky factorization for SPD matrices (exploiting symmetry for half storage and no pivoting), make the library safe for concurrent use, and add sparse matrix-matrix multiply and CSR/CSC export as foundational infrastructure for future sprints.

### Prerequisites from Sprint 3

- Fill-reducing reordering (Cholesky on sparse SPD matrices benefits significantly from AMD/RCM)
- Condition number estimation (useful for detecting non-SPD or near-singular matrices before Cholesky)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Cholesky factorization | Implement `sparse_cholesky_factor()` for symmetric positive-definite matrices. Store only lower triangle. Implement `sparse_cholesky_solve()` (forward/back-sub on L and L^T). Detect non-SPD during factorization (negative/zero pivot). Integrate with fill-reducing reordering. Test against SPD SuiteSparse matrices (nos4, bcsstk04). | 40 hrs |
| 2 | Thread safety | Make pool allocator thread-safe via thread-local pools or mutex protection. Make `sparse_errno()` thread-local (already `_Thread_local`). Document thread-safety guarantees (e.g., read-only sharing of factored matrices is safe, concurrent mutation is not). Add concurrent-access stress tests. Review all global/static state. | 28 hrs |
| 3 | CSR/CSC export | Implement `sparse_to_csr()` and `sparse_to_csc()` to convert the orthogonal linked-list format to compressed sparse row/column arrays. Implement `sparse_from_csr()` and `sparse_from_csc()` for import. Needed for interoperability and efficient column operations in QR (Sprint 6). | 24 hrs |
| 4 | Sparse matrix-matrix multiply | Implement `sparse_matmul(A, B, &C)` computing C = A*B. Row-of-C = linear combination of rows of B, weighted by A's entries. Handle dimension mismatch. Needed for QR (Sprint 6) and SVD (Sprint 8). | 24 hrs |

### Deliverables

- `sparse_cholesky_factor()` and `sparse_cholesky_solve()` in public API
- Cholesky exploits symmetry: stores only lower triangle, ~2x memory savings
- Non-SPD detection with clear error code (`SPARSE_ERR_NOT_SPD`)
- Cholesky integrates with AMD/RCM reordering from Sprint 3
- Thread-local or mutex-protected pool allocator
- Documented thread-safety contract in API headers
- Concurrent stress tests pass under TSan (Thread Sanitizer)
- `sparse_to_csr()`, `sparse_to_csc()`, `sparse_from_csr()`, `sparse_from_csc()` in public API
- `sparse_matmul(A, B, &C)` in public API with tests
- All existing LU tests remain passing

**Total estimate:** ~116 hours

---

## Sprint 5: Iterative Solvers & Preconditioning

**Duration:** 14 days (~120 hours)

**Goal:** Implement Krylov subspace iterative solvers (CG, GMRES) with ILU and Cholesky preconditioning, and add parallel SpMV. These solvers handle larger systems where direct methods are too expensive, and form the iterative backbone needed for QR and SVD convergence loops.

### Prerequisites from Sprint 4

- Cholesky factorization (used as a preconditioner for CG on SPD systems)
- Sparse matrix-matrix multiply (needed for forming preconditioner products)
- Thread safety (parallel SpMV requires safe concurrent reads)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Conjugate Gradient (CG) solver | Implement `sparse_solve_cg()` for SPD systems. Standard CG with optional preconditioner callback. Convergence monitoring via relative residual. Support user-supplied max iterations and tolerance. Test on SPD SuiteSparse matrices (nos4, bcsstk04). | 28 hrs |
| 2 | GMRES solver | Implement `sparse_solve_gmres()` for general (unsymmetric) systems. Restarted GMRES(k) with Arnoldi process and Givens rotations. Support left preconditioning. Test on unsymmetric SuiteSparse matrices (west0067, steam1, orsirr_1). Compare convergence with and without ILU preconditioning. | 36 hrs |
| 3 | Incomplete LU (ILU) preconditioner | Implement `sparse_ilu_factor()` — ILU(0) factorization (LU with no additional fill-in beyond the original sparsity pattern). Implement `sparse_ilu_solve()` for applying the preconditioner. Integrate as preconditioner for CG and GMRES. | 28 hrs |
| 4 | Parallel SpMV | Add OpenMP parallelization to `sparse_matvec()` via row-wise partitioning. Add `#pragma omp parallel for` with appropriate scheduling. Benchmark speedup on multi-core systems. Gate behind compile-time flag (`-DSPARSE_OPENMP`). Verify correctness under TSan. | 20 hrs |

### Deliverables

- `sparse_solve_cg()` for SPD systems with optional preconditioner
- `sparse_solve_gmres()` for general systems with restart and preconditioning
- `sparse_ilu_factor()` / `sparse_ilu_solve()` ILU(0) preconditioner
- Parallel SpMV with OpenMP (optional compile flag)
- Convergence benchmarks: CG vs direct, GMRES vs LU, preconditioned vs unpreconditioned
- All existing direct solver tests remain passing

**Total estimate:** ~112 hours

---

## Sprint 6: Sparse QR Decomposition

**Duration:** 14 days (~130 hours)

**Goal:** Implement full-featured sparse QR factorization using Householder reflections on the orthogonal linked-list structure. QR handles rectangular and rank-deficient matrices that LU cannot, and is the foundation for least-squares solving and SVD.

### Prerequisites from Sprint 4

- CSR/CSC export (efficient column access patterns for Householder reflections)
- Sparse matrix-matrix multiply (for applying Householder reflectors as outer products)

### Prerequisites from Sprint 5

- Iterative solvers (CG/GMRES can be used as alternatives for large least-squares problems; convergence infrastructure reused)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Householder reflections on sparse columns | Implement sparse Householder vector computation: given a sparse column vector x, compute v and beta such that (I - beta*v*v^T)*x = ||x||*e_1. Handle sparse fill-in efficiently — Householder reflections generally densify columns, so maintain a dense working column with sparse writeback. | 24 hrs |
| 2 | Column-pivoted QR factorization | Implement `sparse_qr_factor()` with column pivoting for rank-revealing QR: A*P = Q*R. Use column norms for pivot selection (greedy largest-norm pivot). Store R in upper-triangular sparse matrix. Store Q implicitly as a sequence of Householder reflectors (v vectors + beta scalars). Track column permutation. | 36 hrs |
| 3 | Q application and extraction | Implement `sparse_qr_apply_q()` to apply Q (or Q^T) to a vector or matrix without forming Q explicitly: Q*x = (I - beta_k*v_k*v_k^T)*...*(I - beta_1*v_1*v_1^T)*x. Implement `sparse_qr_form_q()` to explicitly form the Q matrix (for diagnostics/testing, not recommended for large matrices). | 20 hrs |
| 4 | Least-squares solver | Implement `sparse_qr_solve()` for overdetermined systems (m > n): minimize ||Ax - b||_2. Steps: Q^T*b via Householder application, then back-substitute with R. Handle rank deficiency via column pivoting (truncate R at numerical rank). Return residual norm. | 16 hrs |
| 5 | QR integration with reordering | Integrate fill-reducing column reordering (COLAMD-style or reuse AMD on A^T*A pattern) as optional pre-processing to reduce fill-in in R. Benchmark R fill-in with and without reordering. Test on rectangular SuiteSparse matrices. | 16 hrs |
| 6 | Rank estimation and null space | Implement `sparse_qr_rank()` to estimate numerical rank from R diagonal (using tolerance relative to ||A||). Implement `sparse_qr_nullspace()` to extract null-space basis vectors from the trailing columns of Q corresponding to zero/tiny R diagonals. | 12 hrs |

### Deliverables

- `sparse_qr_factor()` with column pivoting (A*P = Q*R)
- Q stored implicitly as Householder reflectors (memory-efficient)
- `sparse_qr_apply_q()` and `sparse_qr_form_q()` for Q operations
- `sparse_qr_solve()` for least-squares on overdetermined systems
- `sparse_qr_rank()` for numerical rank estimation
- `sparse_qr_nullspace()` for null-space basis extraction
- Fill-reducing column reordering integration
- Works on rectangular matrices (m != n)
- Handles rank-deficient matrices gracefully
- Comprehensive tests on square, tall, wide, and rank-deficient matrices
- Benchmark data on SuiteSparse matrices

**Total estimate:** ~124 hours

---

## Sprint 7: QR Applications & Eigenvalue Infrastructure

**Duration:** 14 days (~120 hours)

**Goal:** Build QR-based applications (iterative refinement, economy QR, updating) and lay the eigenvalue computation groundwork needed for SVD. Implement the QR algorithm for eigenvalues of symmetric tridiagonal matrices, which is the core iteration inside SVD.

### Prerequisites from Sprint 6

- Sparse QR factorization (foundation for all items in this sprint)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Economy (thin) QR | Implement economy QR for m >> n: only compute the first n columns of Q and the n×n upper portion of R. Significant memory savings for tall-skinny matrices. Update `sparse_qr_factor()` with an `economy` flag. | 16 hrs |
| 2 | QR iterative refinement | Implement `sparse_qr_refine()` analogous to `sparse_lu_refine()`: compute residual r = b - A*x, solve correction via QR, update x. Useful for improving least-squares solutions on ill-conditioned systems. | 12 hrs |
| 3 | Symmetric tridiagonal QR algorithm | Implement the implicit QR algorithm with Wilkinson shifts for computing eigenvalues of symmetric tridiagonal matrices. This is a dense algorithm on the tridiagonal (stored as two arrays: diagonal + subdiagonal). Used as the inner loop of SVD. Implement deflation when subdiagonal entries converge to zero. | 28 hrs |
| 4 | Bidiagonal reduction | Implement Householder bidiagonalization: reduce a general m×n matrix to upper bidiagonal form B = U^T * A * V using alternating left and right Householder reflections. Store U and V implicitly as Householder sequences. This is the first phase of SVD. | 28 hrs |
| 5 | Dense vector/matrix utilities | Add dense matrix utilities needed by eigenvalue/SVD code: dense matrix create/free/multiply, Givens rotations, and 2×2 symmetric eigenvalue solver. These are small dense operations used inside the tridiagonal QR iteration. | 16 hrs |
| 6 | Sparse transpose | Implement `sparse_transpose()` to compute A^T as a new matrix. Needed for forming A^T*A in SVD and for QR on A^T. | 12 hrs |

### Deliverables

- Economy QR for tall-skinny matrices
- `sparse_qr_refine()` for iterative least-squares improvement
- Tridiagonal QR eigenvalue solver (dense, for SVD inner loop)
- Householder bidiagonalization (for SVD first phase)
- `sparse_transpose()` in public API
- Dense matrix utilities for small inner computations
- All QR and LU tests remain passing

**Total estimate:** ~112 hours

---

## Sprint 8: Sparse SVD

**Duration:** 14 days (~130 hours)

**Goal:** Implement full-featured sparse SVD (Singular Value Decomposition) on the orthogonal linked-list structure. SVD computes A = U*Σ*V^T and is the most powerful matrix decomposition, enabling rank estimation, pseudoinverse, low-rank approximation, and principal component analysis.

### Prerequisites from Sprint 7

- Bidiagonal reduction (first phase of SVD)
- Tridiagonal QR algorithm (inner iteration of SVD, applied to bidiagonal)
- Dense vector/matrix utilities (Givens rotations, 2×2 eigensolvers)
- Sparse transpose (needed for A^T*A approach and verification)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Golub-Kahan bidiagonalization | Adapt the Householder bidiagonalization from Sprint 7 into the Golub-Kahan variant suitable for SVD: produce B, U, V such that A = U*B*V^T where B is upper bidiagonal. Handle rectangular matrices (m > n and m < n). Store U, V implicitly or explicitly based on user request. | 24 hrs |
| 2 | Implicit QR SVD iteration | Implement the Golub-Kahan SVD step (implicit QR on B^T*B without forming it): chase a bulge down the bidiagonal using Givens rotations, accumulating rotations into U and V. Implement Wilkinson shift selection from the trailing 2×2 of B^T*B. Implement deflation (split bidiagonal when superdiagonal converges to zero). | 32 hrs |
| 3 | Full SVD driver | Implement `sparse_svd()` as the top-level driver: bidiagonalize → iterate to convergence → extract singular values and optional U, V. Support options: compute singular values only, thin SVD (economy U/V), or full SVD. Handle convergence failure (max iterations). Sort singular values in descending order. | 20 hrs |
| 4 | Truncated/partial SVD | Implement `sparse_svd_partial()` to compute only the k largest singular values and their corresponding singular vectors, without computing the full decomposition. Use Lanczos bidiagonalization (iterative, builds up the bidiagonal incrementally) for the partial case. Much more efficient than full SVD when k << min(m,n). | 28 hrs |
| 5 | SVD applications | Implement `sparse_svd_rank()` for numerical rank estimation (count singular values above tolerance). Implement `sparse_pinv()` for pseudoinverse via SVD: A^+ = V*Σ^+*U^T. Implement `sparse_svd_lowrank()` for best rank-k approximation. Test on rank-deficient and ill-conditioned matrices. | 20 hrs |

### Deliverables

- `sparse_svd()` — full SVD: A = U*Σ*V^T
- Singular values only, thin SVD, and full SVD modes
- `sparse_svd_partial()` — compute k largest singular values (Lanczos-based)
- `sparse_svd_rank()` — numerical rank estimation
- `sparse_pinv()` — pseudoinverse via SVD
- `sparse_svd_lowrank()` — best rank-k approximation
- Works on rectangular matrices (m != n)
- Handles rank-deficient matrices
- Convergence benchmarks on SuiteSparse matrices
- All existing tests remain passing

**Total estimate:** ~124 hours

---

## Sprint 9: Performance, Polish & Packaging

**Duration:** 14 days (~100 hours)

**Goal:** Optimize performance across the library, add block operations for cache efficiency, improve documentation and examples, and package the library for external use.

### Prerequisites from Sprint 8

- All factorizations complete (LU, Cholesky, QR, SVD)
- Thread safety and parallel SpMV in place

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Block LU factorization | Exploit dense subblocks within the sparse structure for better cache performance. Detect dense submatrices during factorization and use BLAS-like dense kernels for those blocks. Benchmark improvement on matrices with dense substructure. | 28 hrs |
| 2 | Performance profiling & optimization | Profile all factorizations on large SuiteSparse matrices. Identify and optimize hot paths (pool allocation, node traversal, pivot search). Consider SIMD for dense inner loops. Target: ≥1.5x speedup on orsirr_1 factorization. | 24 hrs |
| 3 | Comprehensive examples & tutorials | Write standalone example programs: basic solve, least-squares, SVD low-rank, preconditioned iterative solve. Add a tutorial document walking through common use cases. | 16 hrs |
| 4 | API documentation generator | Add Doxygen configuration. Generate HTML API reference from header comments. Verify all public functions are documented. | 8 hrs |
| 5 | Packaging & installation | Add `make install` target with configurable prefix. Add pkg-config `.pc` file. Add CMake `find_package` support. Write installation instructions for Linux, macOS, and Windows (MSVC). | 16 hrs |
| 6 | Final test hardening | Add fuzz testing for Matrix Market parser. Add property-based tests (random matrices: factor → solve → verify residual). Achieve ≥95% line coverage on library source. | 8 hrs |

### Deliverables

- Block LU for cache-efficient dense subblock handling
- Measurable performance improvements on benchmark suite
- Standalone example programs and tutorial document
- Doxygen-generated API reference
- `make install`, pkg-config, and CMake integration
- Fuzz and property-based tests for robustness
- Production-ready library packaging

**Total estimate:** ~100 hours

---

## Dependency Graph

```
Sprint 2 (Hardening)         ← foundation
    │
Sprint 3 (Reordering)       ← needs norms, tolerance, matrices
    │
Sprint 4 (Cholesky/Threads/SpMM/CSR) ← needs reordering, condest
    │
    ├── Sprint 5 (Iterative Solvers)  ← needs Cholesky, SpMM, threads
    │       │
    │       └── Sprint 6 (QR)         ← needs CSR/CSC, SpMM, iterative infra
    │               │
    │               └── Sprint 7 (QR Apps/Eigenvalues) ← needs QR
    │                       │
    │                       └── Sprint 8 (SVD) ← needs bidiag, tridiag QR
    │
    └── Sprint 9 (Performance/Polish) ← needs all factorizations complete
```

## Summary

| Sprint | Theme | Duration | Estimate | Key Outputs |
|--------|-------|----------|----------|-------------|
| 2 | Hardening & Arithmetic | 14 days | ~54 hrs | ASan, relative tolerance, sparse add/scale, SuiteSparse matrices |
| 3 | Numerics & Reordering | 14 days | ~64 hrs | Condition estimation, AMD/RCM reordering |
| 4 | Cholesky/Threads/SpMM/CSR | 14 days | ~116 hrs | Cholesky, thread safety, SpMM, CSR/CSC export |
| 5 | Iterative Solvers | 14 days | ~112 hrs | CG, GMRES, ILU preconditioner, parallel SpMV |
| 6 | Sparse QR | 14 days | ~124 hrs | Column-pivoted QR, least-squares, rank estimation |
| 7 | QR Apps & Eigenvalues | 14 days | ~112 hrs | Economy QR, bidiagonalization, tridiagonal QR, transpose |
| 8 | Sparse SVD | 14 days | ~124 hrs | Full/partial SVD, pseudoinverse, low-rank approximation |
| 9 | Performance & Polish | 14 days | ~100 hrs | Block LU, profiling, examples, packaging |

**Total across Sprints 2–9:** 112 days (~806 hours)
