# Project Plan: linalg_sparse_orthogonal -- Sprints 11-20 (Epic 2)

Based on findings from the Codex review (`reviews/review-codex-2026-04-06.md`) and Claude review (`reviews/review-claude-2026-04-06.md`).

---

## Sprint 11: Build System, Tolerance Standardization & Thread Safety Fixes

**Duration:** 14 days (~120 hours)

**Goal:** Fix the most urgent quality and correctness issues identified in both reviews before adding new features. Synchronize build systems, standardize numerical tolerances, and resolve thread safety issues.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | CMake test target sync | Add the 6 missing test targets to CMakeLists.txt (test_cholesky, test_csr, test_matmul, test_reorder, test_sprint4_integration, test_threads). Add a CI job that builds and tests with CMake to prevent future drift. | 8 hrs |
| 2 | Tolerance standardization | Replace all hardcoded absolute tolerances (1e-30 in Cholesky/ILU, 1e-300 in CSR solve) with norm-relative tolerances. Add internal `compute_rel_tol(matrix_norm, user_tol)` helper. Audit every factorization and solve path for consistency. Update documentation. | 32 hrs |
| 3 | Thread safety fixes | Fix `cached_norm` data race: use `_Atomic` on cached fields or document the limitation. Audit all shared mutable state. Update README thread safety table to accurately reflect guarantees. Add TSan CI job. | 20 hrs |
| 4 | Factored-state flag | Add `factored` field to SparseMatrix struct. Set in `sparse_lu_factor()` / `sparse_cholesky_factor()`, check in solve functions. Return `SPARSE_ERR_BADARG` for solve-before-factor. | 16 hrs |
| 5 | ILU/ILUT documentation | Document identity-permutation precondition in public headers. Document all undocumented preconditions across the API (QR requires unfactored matrix, etc.). | 8 hrs |
| 6 | Version macro generation | Generate `sparse_version_generated.h` from VERSION file at build time (CMake `configure_file`, Makefile `sed`). Remove hardcoded version macros from `sparse_types.h`. | 10 hrs |
| 7 | 32-bit index documentation | Document `idx_t` as int32_t in README known limitations. Add SIZE_MAX/INT32_MAX rationale to sparse_types.h. | 4 hrs |

### Deliverables

- CMake and Makefile build systems produce identical test coverage
- All solvers use consistent norm-relative tolerance strategy
- Thread safety claims in README match actual guarantees
- Solve-before-factor detected and rejected at runtime
- All public API preconditions documented in headers
- Version managed from single source with no manual sync
- 32-bit index limitation documented

**Total estimate:** ~98 hours

---

## Sprint 12: Sparse LDL^T Factorization (Symmetric Indefinite)

**Duration:** 14 days (~160 hours)

**Goal:** Add the highest-priority missing factorization: sparse LDL^T with Bunch-Kaufman symmetric pivoting for symmetric indefinite systems (KKT, saddle-point, constrained optimization).

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | LDL^T data structures | Define `sparse_ldlt_t` result struct (permutation, D diagonal/2x2 blocks, L unit lower triangular). Design public API in `sparse_ldlt.h`. | 12 hrs |
| 2 | Bunch-Kaufman pivoting | Implement symmetric pivoting strategy: 1x1 and 2x2 pivots with growth-factor control. Handle diagonal block bookkeeping. | 32 hrs |
| 3 | Sparse LDL^T elimination | Implement the elimination loop operating on linked-list format. Store L below diagonal, D as diagonal/2x2 blocks. Support fill-reducing reordering (AMD/RCM) as preprocessing. | 40 hrs |
| 4 | LDL^T solve | Forward substitution (L), diagonal solve (D, handling 2x2 blocks), backward substitution (L^T), and permutation application. | 20 hrs |
| 5 | LDL^T tests | Test on symmetric indefinite matrices: KKT systems, saddle-point matrices, matrices with negative eigenvalues. Compare with LU on same systems. SuiteSparse symmetric indefinite matrices. | 24 hrs |
| 6 | Integration and documentation | Add to README, update API overview table, add example usage. Integration tests with iterative refinement. | 16 hrs |

### Deliverables

- `sparse_ldlt_factor()` and `sparse_ldlt_solve()` in public API
- Bunch-Kaufman 1x1/2x2 symmetric pivoting
- AMD/RCM reordering support
- Tests on KKT and saddle-point systems
- Documented in README and headers

**Total estimate:** ~144 hours

---

## Sprint 13: Incomplete Cholesky Preconditioner & MINRES Solver

**Duration:** 14 days (~136 hours)

**Goal:** Add the two highest-value iterative solver extensions: incomplete Cholesky (IC(0)) preconditioning for SPD systems and MINRES for symmetric indefinite systems.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Incomplete Cholesky IC(0) | Implement no-fill incomplete Cholesky: L*L^T approximation preserving the sparsity pattern of A. Store as sparse_ilu_t (reuse L/U struct with L^T = U). | 28 hrs |
| 2 | IC(0) preconditioner callback | Implement `sparse_ic_precond()` compatible with `sparse_precond_fn`. Forward-backward substitution with L and L^T. | 8 hrs |
| 3 | MINRES solver | Implement Lanczos-based MINRES for symmetric (possibly indefinite) systems. Support preconditioning via callback. Return iterations, residual, convergence flag. | 40 hrs |
| 4 | Block MINRES | Multi-RHS MINRES variant with per-column convergence tracking, consistent with block CG/GMRES API pattern. | 20 hrs |
| 5 | IC(0) + MINRES integration | Test IC(0)-preconditioned CG on SPD systems (compare with ILU(0)). Test MINRES on symmetric indefinite systems with IC and LDL^T preconditioners. | 24 hrs |
| 6 | Documentation and benchmarks | Document IC(0) and MINRES in README, headers, INSTALL.md. Benchmark IC(0) vs ILU(0) on SPD SuiteSparse matrices. | 16 hrs |

### Deliverables

- `sparse_ic_factor()`, `sparse_ic_solve()`, `sparse_ic_precond()` in public API
- `sparse_solve_minres()` and `sparse_minres_solve_block()` in public API
- IC(0) tested as CG preconditioner on SPD systems
- MINRES tested on symmetric indefinite systems
- Benchmark comparison: IC(0) vs ILU(0) on SPD matrices

**Total estimate:** ~136 hours

---

## Sprint 14: Symbolic Analysis / Numeric Factorization Split

**Duration:** 14 days (~152 hours)

**Goal:** Separate symbolic analysis from numeric factorization for LU, Cholesky, and LDL^T. This enables repeated numeric refactorization on the same sparsity pattern without redoing ordering and symbolic work.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Elimination tree computation | Compute the elimination tree (etree) from the sparsity pattern. Store parent pointers and postordering. This is the foundation for symbolic factorization. | 24 hrs |
| 2 | Symbolic factorization for Cholesky | Compute the nonzero structure of L from the etree without performing numeric work. Return column counts, row indices, and total nnz. | 24 hrs |
| 3 | Symbolic factorization for LU | Compute upper bounds on L and U sparsity structure using the etree and column dependency graph. Pre-allocate storage. | 28 hrs |
| 4 | Analysis object API | Design `sparse_analysis_t` struct holding etree, column counts, permutation, and symbolic factor structure. API: `sparse_analyze(A, opts, &analysis)` -> `sparse_factor_numeric(A, analysis, &factors)`. | 20 hrs |
| 5 | Numeric refactorization | Implement `sparse_refactor_numeric()` that reuses the symbolic structure from a prior analysis to factor a new matrix with the same sparsity pattern. Validate pattern match. | 24 hrs |
| 6 | Backward compatibility | Keep existing one-shot `sparse_lu_factor()` and `sparse_cholesky_factor()` APIs working by internally calling analyze + factor. No breaking changes. | 16 hrs |
| 7 | Tests and documentation | Test repeated refactorization (same pattern, different values). Verify memory usage is predictable. Benchmark symbolic-once vs. repeated full factorization. | 16 hrs |

### Deliverables

- `sparse_analyze()` computes etree and symbolic structure
- `sparse_factor_numeric()` performs numeric-only factorization
- `sparse_refactor_numeric()` reuses symbolic structure
- Existing one-shot APIs still work (backward compatible)
- Repeated factorization benchmark showing speedup

**Total estimate:** ~152 hours

---

## Sprint 15: COLAMD Ordering & QR Minimum-Norm Least Squares

**Duration:** 14 days (~140 hours)

**Goal:** Upgrade the ordering stack with COLAMD for unsymmetric/QR problems and add minimum-norm least-squares solves for underdetermined systems.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | COLAMD ordering | Implement column approximate minimum degree ordering for unsymmetric matrices and QR factorization. Operate on the column adjacency graph (A^T*A pattern) without forming it explicitly. | 40 hrs |
| 2 | Integrate COLAMD with QR | Add `SPARSE_REORDER_COLAMD` to the reorder enum. Wire into `sparse_qr_factor_opts()` as the default column ordering for QR. Benchmark fill reduction vs current AMD proxy. | 16 hrs |
| 3 | QR minimum-norm solve | Implement minimum-norm least-squares for underdetermined systems (m < n): compute Q*R factorization of A^T, then solve via R^{-T} * Q^T * b. Return the minimum 2-norm solution. | 32 hrs |
| 4 | QR solve API update | Add `sparse_qr_solve_minnorm()` or a flag in the QR solve options to request minimum-norm. Update `sparse_qr_solve()` documentation to clearly distinguish basic vs minimum-norm solutions. | 12 hrs |
| 5 | Rank-revealing improvements | Improve rank detection diagnostics: expose the R diagonal, add rank-deficiency warnings, document threshold selection guidance. | 16 hrs |
| 6 | Tests and documentation | Test COLAMD on SuiteSparse unsymmetric matrices. Test minimum-norm on underdetermined systems with known solutions. Update README. | 24 hrs |

### Deliverables

- `sparse_reorder_colamd()` in public API
- COLAMD integrated as default QR column ordering
- `sparse_qr_solve_minnorm()` for underdetermined systems
- Rank-revealing diagnostics improved
- Fill reduction benchmarks: COLAMD vs AMD on unsymmetric matrices

**Total estimate:** ~140 hours

---

## Sprint 16: BiCGSTAB Solver & Iterative Solver Hardening

**Duration:** 14 days (~128 hours)

**Goal:** Add BiCGSTAB for nonsymmetric systems where restarted GMRES is a poor fit, and harden the iterative solver framework with better convergence diagnostics, stagnation detection, and breakdown handling.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | BiCGSTAB solver | Implement BiCG Stabilized method for general nonsymmetric systems. Support left preconditioning via callback. Return iterations, residual, convergence flag. | 32 hrs |
| 2 | Block BiCGSTAB | Multi-RHS variant with per-column convergence, consistent with block CG/GMRES pattern. | 16 hrs |
| 3 | Matrix-free BiCGSTAB | User-supplied matvec callback variant, following the CG/GMRES matrix-free pattern. | 8 hrs |
| 4 | Stagnation detection | Add stagnation detection to CG, GMRES, MINRES, and BiCGSTAB: if residual fails to decrease for N consecutive iterations, report `SPARSE_ERR_NOT_CONVERGED` early with a stagnation flag in the result struct. | 20 hrs |
| 5 | Convergence diagnostics | Add optional residual history recording (caller-provided array). Add verbose callback option for custom progress reporting. | 16 hrs |
| 6 | Breakdown handling | Audit all iterative solvers for graceful handling of breakdown (zero denominators in CG, lucky breakdown in GMRES, BiCGSTAB rho=0). Document breakdown behavior. | 20 hrs |
| 7 | Tests and documentation | Compare BiCGSTAB vs GMRES on nonsymmetric SuiteSparse matrices. Test stagnation detection. Update README iterative solver section. | 16 hrs |

### Deliverables

- `sparse_solve_bicgstab()`, block, and matrix-free variants
- Stagnation detection in all iterative solvers
- Optional residual history recording
- Breakdown handling audited and documented
- BiCGSTAB vs GMRES comparison benchmarks

**Total estimate:** ~128 hours

---

## Sprint 17: CSR/CSC Numeric Backend for Cholesky and LDL^T

**Duration:** 14 days (~152 hours)

**Goal:** Extend the CSR working-format strategy (proven in Sprint 10 for LU) to Cholesky and LDL^T, making compressed formats the primary numeric backend for all direct solvers while keeping the linked list as the mutable front end.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | CSC format for Cholesky | Implement CSC (compressed sparse column) working format for Cholesky, exploiting column-oriented access patterns in L*L^T factorization. Convert from linked-list, factor in CSC, convert back. | 36 hrs |
| 2 | CSC Cholesky elimination | Column-by-column Cholesky factorization on CSC arrays. Scatter-gather pattern for column updates. Pre-allocated storage from symbolic analysis (Sprint 14). | 32 hrs |
| 3 | CSC LDL^T elimination | Adapt the CSC Cholesky kernel for LDL^T with 1x1/2x2 diagonal blocks. Handle symmetric pivoting in CSC format. | 28 hrs |
| 4 | Supernodal detection for Cholesky | Extend the dense block detection from CSR LU to CSC Cholesky: identify supernodes (groups of columns with identical nonzero structure) and factor them with dense BLAS-like kernels. | 24 hrs |
| 5 | Benchmarks and validation | Benchmark CSC Cholesky/LDL^T vs linked-list on SuiteSparse SPD and symmetric indefinite matrices. Validate residuals match. Target >= 2x speedup. | 16 hrs |
| 6 | Documentation | Update README performance section. Document CSC working format in sparse_lu_csr.h (or new sparse_chol_csc.h). | 16 hrs |

### Deliverables

- CSC Cholesky factorization with >= 2x speedup over linked-list path
- CSC LDL^T factorization
- Supernodal detection and dense kernel optimization for Cholesky
- Benchmark results on SPD and symmetric indefinite matrices

**Total estimate:** ~152 hours

---

## Sprint 18: Sparse Eigensolvers (Lanczos & LOBPCG)

**Duration:** 14 days (~144 hours)

**Goal:** Add sparse eigenpair routines built on the Lanczos and LOBPCG infrastructure, moving beyond the current tridiagonal QR kernel to full-featured sparse eigenvalue computation.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Symmetric Lanczos eigensolver | Implement thick-restart Lanczos for computing k largest/smallest eigenvalues and eigenvectors of symmetric matrices. Reorthogonalization for numerical stability. | 40 hrs |
| 2 | Shift-invert Lanczos | Add shift-invert mode for interior eigenvalues: solve (A - sigma*I)^{-1} * x using LU or LDL^T factorization inside the Lanczos iteration. | 20 hrs |
| 3 | LOBPCG solver | Implement Locally Optimal Block Preconditioned Conjugate Gradient for symmetric eigenvalue problems. Support preconditioning and block computation of multiple eigenpairs. | 36 hrs |
| 4 | Eigenvalue API design | Define `sparse_eigs_t` result struct (eigenvalues, eigenvectors, convergence info). API: `sparse_eigs_sym(A, k, which, opts, &result)` with `which` = largest/smallest/nearest_sigma. | 12 hrs |
| 5 | Tests and validation | Test on diagonal matrices (exact eigenvalues known), tridiagonal matrices, SuiteSparse SPD matrices. Validate against existing SVD (eigenvalues of A^T*A = singular values squared). | 20 hrs |
| 6 | Documentation and examples | Document eigenvalue API in README, add example program, update API table. | 16 hrs |

### Deliverables

- `sparse_eigs_sym()` with Lanczos backend for k largest/smallest eigenvalues
- Shift-invert mode for interior eigenvalues
- `sparse_eigs_lobpcg()` for preconditioned block eigenvalue computation
- Tests validating against known eigenvalues and SVD consistency

**Total estimate:** ~144 hours

---

## Sprint 19: Nested Dissection Ordering & Large-Scale Infrastructure

**Duration:** 14 days (~136 hours)

**Goal:** Add nested dissection ordering for large sparse problems and infrastructure for handling matrices beyond the current test scale, including better memory management and progress reporting.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Graph partitioning | Implement a vertex separator algorithm for sparse graphs (multilevel bisection or spectral partitioning). This is the core building block for nested dissection. | 32 hrs |
| 2 | Nested dissection ordering | Implement recursive nested dissection: partition the graph, order interior nodes of each partition first, then separator nodes. Produces fill-reducing orderings superior to AMD for 2D/3D PDE meshes. | 28 hrs |
| 3 | Add SPARSE_REORDER_ND to enum | Wire nested dissection into the existing reorder infrastructure. Benchmark against AMD/RCM on large SuiteSparse matrices. | 12 hrs |
| 4 | Quotient-graph AMD | Replace the current bitset-based AMD (O(n^3/64) time, O(n^2/64) memory) with a quotient-graph implementation that operates in O(nnz) memory. This removes the current scaling bottleneck for AMD on large matrices. | 32 hrs |
| 5 | Progress/cancel callbacks | Add optional progress callback to long-running factorization and iterative solve routines. Allow cancellation via callback return value. | 16 hrs |
| 6 | Tests and benchmarks | Test nested dissection on 2D/3D mesh matrices. Benchmark AMD (quotient-graph) vs AMD (bitset) on large matrices. Test progress callbacks. | 16 hrs |

### Deliverables

- `sparse_reorder_nd()` nested dissection ordering
- Quotient-graph AMD replacing bitset AMD for O(nnz) memory usage
- Progress/cancel callbacks for long-running operations
- Ordering benchmarks on large matrices (fill-in, memory, time)

**Total estimate:** ~136 hours

---

## Sprint 20: SVD Improvements, CI Hardening & Epic 2 Wrap-Up

**Duration:** 14 days (~128 hours)

**Goal:** Address remaining review findings: fix the dense-in-disguise SVD paths, add Windows/macOS CI, improve the sparse low-rank approximation, and close out Epic 2 with final documentation and validation.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Sparse low-rank without dense accumulator | Rewrite `sparse_svd_lowrank_sparse()` to build the rank-k approximation using outer product accumulation directly into sparse output, eliminating the current m*n dense intermediate. | 24 hrs |
| 2 | Full SVD U/V output beyond economy mode | Extend full SVD to optionally output complete (non-economy) U and V^T when requested. Currently only economy mode is supported for U/V. | 20 hrs |
| 3 | Windows CI with CMake | Add GitHub Actions job for Windows/MSVC using CMake. Fix any remaining portability issues (conditional test_fuzz exclusion is already done). | 16 hrs |
| 4 | macOS CI job | Add GitHub Actions job for macOS. Test both Apple Clang and Homebrew GCC. Verify coverage and packaging scripts work. | 12 hrs |
| 5 | API accessor error reporting | Add `sparse_get_err()` variant that returns error codes alongside values, or document the silent-zero-on-error contract explicitly in all accessor headers. | 12 hrs |
| 6 | Final integration testing | Full regression under all sanitizers, all platforms. Cross-feature tests for new Sprint 11-20 features. Benchmark suite on representative matrix collection. | 20 hrs |
| 7 | Epic 2 retrospective and documentation | Update README with all new APIs (LDL^T, IC, MINRES, BiCGSTAB, eigensolvers, COLAMD, ND). Write Epic 2 retrospective. Update INSTALL.md for new platforms. | 24 hrs |

### Deliverables

- Sparse low-rank approximation without dense intermediate
- Full U/V SVD output option
- CI on Windows (MSVC) and macOS (Clang + GCC)
- All new APIs documented in README
- Epic 2 retrospective with metrics and assessment

**Total estimate:** ~128 hours

---

## Summary

| Sprint | Title | Key Deliverables | Estimate |
|--------|-------|-----------------|----------|
| 11 | Build System, Tolerances & Thread Safety | CMake sync, tolerance standardization, thread safety fixes, factored-state flag | 98 hrs |
| 12 | Sparse LDL^T Factorization | Symmetric indefinite direct solver with Bunch-Kaufman pivoting | 144 hrs |
| 13 | Incomplete Cholesky & MINRES | IC(0) preconditioner, MINRES solver for symmetric indefinite systems | 136 hrs |
| 14 | Symbolic/Numeric Factorization Split | Elimination tree, symbolic analysis, numeric refactorization | 152 hrs |
| 15 | COLAMD Ordering & QR Min-Norm | COLAMD for QR, minimum-norm least squares for underdetermined systems | 140 hrs |
| 16 | BiCGSTAB & Iterative Hardening | BiCGSTAB solver, stagnation detection, convergence diagnostics | 128 hrs |
| 17 | CSR/CSC Numeric Backend | CSC Cholesky and LDL^T with supernodal optimization | 152 hrs |
| 18 | Sparse Eigensolvers | Lanczos, shift-invert, LOBPCG for symmetric eigenvalue problems | 144 hrs |
| 19 | Nested Dissection & Scale | Nested dissection ordering, quotient-graph AMD, progress callbacks | 136 hrs |
| 20 | SVD Fixes, CI & Wrap-Up | Sparse low-rank fix, full SVD, Windows/macOS CI, retrospective | 128 hrs |

**Total Epic 2 estimate:** ~1,358 hours across 10 sprints (140 days)
