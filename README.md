# linalg_sparse_orthogonal

A C library for sparse matrices using the **orthogonal linked-list** (cross-linked) representation, with direct and iterative linear system solvers.

## Features

### Core Data Structure
- **Orthogonal linked-list storage** — each non-zero is linked into both its row list and column list, enabling efficient row and column traversal
- **Slab pool allocator** with free-list for fast node allocation and reuse

### Direct Solvers
- **LU factorization** with complete or partial pivoting (P·A·Q = L·U)
- **CSR LU factorization** — scatter-gather elimination on compressed sparse row arrays for >=2x speedup on large matrices, with dense subblock detection and in-place dense kernels
- **Block LU solve** — solve A·X = B for multiple right-hand sides simultaneously (`sparse_lu_solve_block`)
- **Cholesky factorization** for symmetric positive-definite matrices (A = L·L^T, ~50% less storage than LU)
- **CSC Cholesky factorization** — column-oriented scatter-gather kernel with fundamental supernode detection, batched supernodal dense kernels (Sprint 18), and transparent size-based dispatch from `sparse_cholesky_factor_opts`; measured up to 4.4× one-shot speedup over the linked-list path on SuiteSparse SPD matrices (n ≤ 14 822), with further gains in the analyze-once / factor-many workflow (Sprint 17 + Sprint 18)
- **LDL^T factorization** with Bunch-Kaufman symmetric pivoting for symmetric indefinite matrices (P·A·P^T = L·D·L^T) — 1x1 and 2x2 pivot blocks, inertia computation, iterative refinement, condition estimation
- **CSC LDL^T factorization** — CSC storage for the L factor + auxiliary D/D_offdiag/pivot_size/perm arrays, scalar triangular + block-diagonal solve path, native Bunch-Kaufman kernel with 1×1 / 2×2 pivots and symmetric swaps (Sprint 18); per-row adjacency index for sparse-row cmod scaling (Sprint 19 Days 8-9); batched supernodal LDL^T mirroring the Cholesky batched path (Sprint 19 Days 10-13).  Native kernel reaches 3.5× factor speedup on bcsstk14 vs linked-list; batched supernodal reaches 6.8× on the same matrix.
- **QR factorization** with column pivoting (A·P = Q·R) — Householder reflections, least-squares, rank estimation, null-space extraction, economy (thin) QR, sparse-mode QR without dense workspace
- **QR minimum-norm solve** for underdetermined systems (m < n) — minimum 2-norm solution via QR of A^T (`sparse_qr_solve_minnorm`)
- **QR rank diagnostics** — R diagonal extraction (`sparse_qr_diag_r`), rank info with condition estimate (`sparse_qr_rank_info`), quick condition estimator (`sparse_qr_condest`)
- **QR iterative refinement** to improve least-squares and minimum-norm solutions
- **Householder bidiagonalization** — reduces A to upper bidiagonal form B = U^T·A·V (SVD preprocessing)
- **Direct solve** via forward/backward substitution with permutation handling
- **Iterative refinement** to improve solution accuracy

### Singular Value Decomposition (SVD)
- **Full SVD** via Golub-Kahan bidiagonalization + implicit QR iteration (A = U·Σ·V^T)
- **Partial/truncated SVD** via Lanczos bidiagonalization — k largest singular values with optional singular vectors
- **Zero-diagonal chase** (Golub & Van Loan §8.6.2) for rank-deficient matrices
- **Condition number** estimation via SVD (`sparse_cond`)
- **Pseudoinverse** via SVD (`sparse_pinv`) — Moore-Penrose A^+
- **Low-rank approximation** — dense (`sparse_svd_lowrank`) and sparse (`sparse_svd_lowrank_sparse`) output
- **Numerical rank** estimation (`sparse_svd_rank`)

### Iterative Solvers
- **Conjugate Gradient (CG)** for SPD systems with optional preconditioning
- **Block CG** — simultaneous CG for multiple RHS with shared SpMV and per-column convergence (`sparse_cg_solve_block`)
- **Restarted GMRES(k)** for general unsymmetric systems with left and right preconditioning
- **Multi-RHS GMRES** — restarted GMRES(k) applied independently per RHS with aggregated reporting (`sparse_gmres_solve_block`)
- **Matrix-free variants** — CG, GMRES, and BiCGSTAB with user-supplied matvec callback (`sparse_solve_cg_mf`, `sparse_solve_gmres_mf`, `sparse_solve_bicgstab_mf`)
- **MINRES** for symmetric (possibly indefinite) systems — Lanczos-based minimum residual with monotonic residual decrease (`sparse_solve_minres`, `sparse_minres_solve_block`)
- **BiCGSTAB** for general nonsymmetric systems — stabilized bi-conjugate gradient with left preconditioning, O(n) storage (`sparse_solve_bicgstab`, `sparse_bicgstab_solve_block`, `sparse_solve_bicgstab_mf`)
- **Stagnation detection** — optional sliding-window residual monitoring across all iterative solvers; early exit when residual stops decreasing (`stagnation_window` in opts)
- **Convergence diagnostics** — optional per-iteration residual history recording (`residual_history` in opts) and user-supplied verbose callback (`sparse_iter_callback_fn`)
- **Breakdown handling** — threshold-based detection and reporting for all solver breakdown conditions (CG p^T*Ap=0, GMRES lucky breakdown, MINRES Lanczos, BiCGSTAB rho=0/omega=0)
- **ILU(0) preconditioner** — incomplete LU with no fill-in, 3-1000× iteration reduction
- **ILUT preconditioner** — ILU with threshold dropping and controlled fill-in, handles zero-diagonal matrices, optional row partial pivoting
- **IC(0) preconditioner** — incomplete Cholesky for SPD systems, symmetric analogue of ILU(0) (`sparse_ic_factor`, `sparse_ic_precond`)

### Eigenvalue Infrastructure
- **Symmetric tridiagonal QR algorithm** — implicit QR with Wilkinson shifts and deflation (eigenvalues via `tridiag_qr_eigenvalues`, eigenpairs via `tridiag_qr_eigenpairs`)
- **2×2 symmetric eigensolver** — numerically stable quadratic formula
- **Dense matrix utilities** — Givens rotations, matrix-matrix/vector multiply

### Sparse Symmetric Eigensolver (Sprints 20-21)
- **`sparse_eigs_sym`** — k extreme or near-sigma eigenpairs of a symmetric sparse matrix.  Three concrete backends, picked transparently by `opts->backend = SPARSE_EIGS_BACKEND_AUTO` per the Sprint 21 Day 10 decision tree (overridable explicitly):
  - **Grow-m Lanczos** (`SPARSE_EIGS_BACKEND_LANCZOS`, Sprint 20) — full MGS reorthogonalization with a growing-subspace outer loop.  Peak memory `O(m_cap · n)`.  AUTO default for `n < SPARSE_EIGS_THICK_RESTART_THRESHOLD` (500).
  - **Wu/Simon thick-restart Lanczos** (`SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`, Sprint 21 Days 1-4) — preserves the converged Ritz subspace in a compact arrowhead basis between restart phases; peak memory `O((k + m_restart) · n)` regardless of total iteration count.  bcsstk14 (n = 1806, k = 5) drops from ~7 MB of `V` (grow-m) to ~565 KB.  AUTO default for `n ≥ 500` when no preconditioner is supplied.
  - **LOBPCG** (`SPARSE_EIGS_BACKEND_LOBPCG`, Sprint 21 Days 7-10) — Knyazev's Locally Optimal Block Preconditioned Conjugate Gradient with block Rayleigh-Ritz over the `[X | W | P]` subspace, BLOPEX-style conditioning guard, and per-column soft-locking.  Plugs the Sprint 13 IC(0) / LDL^T preconditioners in via `opts->precond` (same `sparse_precond_fn` callback the iterative solvers use).  AUTO routes here for `n ≥ SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` (1000) when `opts->precond != NULL` and the block size is at least 4.
- Three `which` modes across all backends — `LARGEST`, `SMALLEST`, `NEAREST_SIGMA` (shift-invert via the Sprint 20 Day 4-6 `sparse_ldlt_factor_opts` AUTO LDL^T dispatch — composes with the CSC supernodal backend on n ≥ `SPARSE_CSC_THRESHOLD` indefinite inputs).
- **Parallel MGS reorthogonalization** (Sprint 21 Days 5-6) — both Lanczos backends parallelise the inner-product / daxpy bodies under `-DSPARSE_OPENMP`, gated on `n ≥ SPARSE_EIGS_OMP_REORTH_MIN_N` (500) so small problems don't pay OMP fork/join overhead.  ~2× speedup at 4 threads on bcsstk14.  The outer `j` loop stays serial — modified Gram-Schmidt's stability bound requires each iteration to see the partially-orthogonalised vector from the previous subtraction (classical Gram-Schmidt parallelises `j` but loses the stability).
- **Ritz-pair output** — optional eigenvectors via `compute_vectors = 1`; Wu/Simon per-pair residuals reported in `result.residual_norm`.
- **Observability** — `result.used_csc_path_ldlt` reports the LDL^T backend chosen on the shift-invert path; `result.peak_basis_size` reports the simultaneously-live `V` columns for memory budgeting; `result.backend_used` reports which backend AUTO actually picked.
- **Preconditioning speedup** — on bcsstk04 (n = 132, cond ≈ 5e6) k = 3 SMALLEST: vanilla LOBPCG saturates the 800-iteration cap with residual ~1e+01, IC(0) preconditioning converges in **62 iterations** at residual 8e-9, LDL^T preconditioning converges in **8 iterations** at residual 3e-9.
- **`bench_eigs`** — permanent benchmark driver (Sprint 21 Day 11) at `benchmarks/bench_eigs.c`; CLI with `--sweep default`, `--compare`, and `--matrix <path>` modes, CSV output, configurable `--repeats`.  Run via `make bench-eigs` for the smoke target.  See `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt` (full sweep) and `docs/planning/EPIC_2/SPRINT_21/bench_day14_compare.txt` (3-backend × 3-precond pivot) for the measured numbers; `benchmarks/README.md` documents the CSV schema.

**Picking a backend** — pass `SPARSE_EIGS_BACKEND_AUTO` (the zero default) and let the library choose: small problems run on grow-m Lanczos; medium-to-large problems route to thick-restart Lanczos for the bounded memory; large problems with a preconditioner route to LOBPCG.  Override with an explicit `opts->backend` when profiling or when the workload differs from the bench-corpus heuristics.

### Matrix Operations
- **Sparse matrix-vector product** (SpMV) with optional OpenMP parallelization
- **Block SpMV** — sparse matrix times dense block Y = A·X for nrhs vectors (`sparse_matvec_block`)
- **Sparse matrix-matrix multiply** — C = A*B via Gustavson's algorithm (`sparse_matmul`)
- **Sparse transpose** — compute A^T as a new matrix (`sparse_transpose`)
- **Matrix arithmetic** — scalar scaling (`sparse_scale`) and addition (`sparse_add`)
- **Infinity norm** with internal caching (`sparse_norminf`)

### Symbolic Analysis & Refactorization
- **Elimination tree** computation via Liu's algorithm with path compression
- **Symbolic Cholesky/LU factorization** — predict exact symbolic structure for Cholesky (upper bound on stored numeric factor when dropping is enabled) or upper-bound sparsity structure for LU, without numeric work
- **Analyze-once, factor-many workflow** — `sparse_analyze()` → `sparse_factor_numeric()` → `sparse_refactor_numeric()` for repeated solves with the same sparsity pattern but different values
- **Column counts** — predict symbolic nnz per column of L for pre-allocation (upper bound on stored numeric counts when dropping is enabled)

### Reordering & Preconditioning
- **Fill-reducing reordering** — Reverse Cuthill-McKee (RCM), Approximate Minimum Degree (AMD), and Column Approximate Minimum Degree (COLAMD) for unsymmetric/QR problems
- **Condition number estimation** — Hager/Higham 1-norm estimator from LU or LDL^T factors, quick R-diagonal estimator from QR

### I/O & Interop
- **Matrix Market I/O** — load and save `.mtx` files (coordinate real general, symmetric, and pattern formats)
- **CSR/CSC export/import** — convert to/from compressed sparse row/column formats

### Quality
- **Thread-safe** — concurrent solves on shared factored matrices, per-matrix pool allocators
- **Parallel SpMV** — OpenMP row-wise parallelization (compile with `-DSPARSE_OPENMP`)
- **errno capture** for I/O errors (`sparse_errno`)

## Building

### With Make (recommended)

```bash
make            # build library
make test       # run all unit tests
make bench      # run benchmarks
make examples   # build standalone example programs
make docs       # generate Doxygen API reference (requires doxygen)
make omp        # build and test with OpenMP-enabled parallel SpMV
make sanitize   # build with undefined-behavior sanitizer
make coverage   # generate lcov coverage report (requires gcc + lcov + bc)
make install    # install to PREFIX (default /usr/local)
make uninstall  # remove installed files
make clean      # remove build artifacts
```

### With CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build .
ctest           # run tests
cmake --install .   # install (supports find_package(Sparse))
```

See [INSTALL.md](INSTALL.md) for detailed cross-platform instructions.

### Compiler Requirements

- C11-compatible compiler (GCC, Clang, etc.)
- Standard math library (`-lm`)

## Quick Start

```c
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include <stdio.h>

int main(void)
{
    /* Create a 3x3 system: A*x = b */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);  sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);  sparse_insert(A, 1, 1, 3.0);  sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);  sparse_insert(A, 2, 2, 4.0);

    double b[] = {5.0, 10.0, 13.0};  /* known solution: x = [1, 2, 3] */
    double x[3];

    /* Factor and solve */
    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_solve(LU, b, x);

    printf("x = [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);

    /* Optional: iterative refinement */
    sparse_lu_refine(A, LU, b, x, 5, 1e-15);

    sparse_free(LU);
    sparse_free(A);
    return 0;
}
```

Compile and link:

```bash
make
cc -Iinclude -o example example.c -Lbuild -lsparse_lu_ortho -lm
```

### Iterative Solver Example

```c
#include "sparse_matrix.h"
#include "sparse_iterative.h"
#include "sparse_ilu.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    /* Load a matrix from Matrix Market file */
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, "matrix.mtx") != SPARSE_OK) {
        fprintf(stderr, "Failed to load matrix\n");
        return 1;
    }
    int n = sparse_rows(A);

    double *b = malloc(n * sizeof(double));
    double *x = calloc(n, sizeof(double));  /* zero initial guess */
    /* ... set up b ... */

    /* ILU(0) preconditioned GMRES */
    sparse_ilu_t ilu;
    if (sparse_ilu_factor(A, &ilu) != SPARSE_OK) {
        fprintf(stderr, "ILU factorization failed\n");
        free(b); free(x); sparse_free(A);
        return 1;
    }

    sparse_gmres_opts_t opts = { .max_iter = 1000, .restart = 50, .tol = 1e-10 };
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts,
                                           sparse_ilu_precond, &ilu, &result);

    if (err == SPARSE_OK)
        printf("Converged in %d iterations, residual = %e\n",
               result.iterations, result.residual_norm);
    else
        printf("Solver returned: %s\n", sparse_strerror(err));

    sparse_ilu_free(&ilu);
    free(b); free(x);
    sparse_free(A);
}
```

## API Overview

| Header | Purpose |
|--------|---------|
| [`sparse_types.h`](include/sparse_types.h) | `idx_t`, error codes (`sparse_err_t`), pivot/reorder strategies, version macros |
| [`sparse_matrix.h`](include/sparse_matrix.h) | Sparse matrix lifecycle, element access, SpMV, block SpMV, Matrix Market I/O |
| [`sparse_lu.h`](include/sparse_lu.h) | LU factorization, solve, block solve, condition estimation, iterative refinement |
| [`sparse_lu_csr.h`](include/sparse_lu_csr.h) | CSR LU working format — conversion, scatter-gather elimination, dense block detection, block solve |
| [`sparse_cholesky.h`](include/sparse_cholesky.h) | Cholesky factorization and solve for SPD matrices |
| [`sparse_ldlt.h`](include/sparse_ldlt.h) | LDL^T factorization with Bunch-Kaufman pivoting for symmetric indefinite matrices |
| [`sparse_analysis.h`](include/sparse_analysis.h) | Symbolic analysis, numeric factorization, refactorization (analyze-once workflow) |
| [`sparse_iterative.h`](include/sparse_iterative.h) | CG, GMRES, MINRES, BiCGSTAB; block CG/GMRES/MINRES; GMRES left/right preconditioning |
| [`sparse_ilu.h`](include/sparse_ilu.h) | ILU(0) and ILUT incomplete factorization preconditioners |
| [`sparse_ic.h`](include/sparse_ic.h) | IC(0) incomplete Cholesky preconditioner for SPD systems |
| [`sparse_qr.h`](include/sparse_qr.h) | Column-pivoted QR factorization, least-squares, rank, null space, refinement |
| [`sparse_dense.h`](include/sparse_dense.h) | Dense matrix utilities, Givens rotations, 2×2 eigensolver, tridiag QR |
| [`sparse_bidiag.h`](include/sparse_bidiag.h) | Householder bidiagonalization (SVD preprocessing) |
| [`sparse_csr.h`](include/sparse_csr.h) | CSR/CSC compressed format conversion |
| [`sparse_reorder.h`](include/sparse_reorder.h) | Fill-reducing reordering (RCM, AMD, COLAMD), permutation, bandwidth |
| [`sparse_svd.h`](include/sparse_svd.h) | SVD, partial SVD, condition number, pseudoinverse, low-rank approximation |
| [`sparse_eigs.h`](include/sparse_eigs.h) | Sparse symmetric eigensolver — Lanczos with growing-m outer loop, shift-invert mode, Ritz pairs |
| [`sparse_vector.h`](include/sparse_vector.h) | Dense vector utilities (norms, axpy, dot product) |

### Key Functions

**Matrix lifecycle:**
- `sparse_create(rows, cols)` — create an empty matrix
- `sparse_free(mat)` — free all memory
- `sparse_copy(mat)` — deep copy

**Element access:**
- `sparse_insert(mat, row, col, val)` — insert or update (inserting 0.0 removes)
- `sparse_get_phys(mat, row, col)` — read at physical index
- `sparse_get(mat, row, col)` / `sparse_set(mat, row, col, val)` — logical (through permutations)

**Solving linear systems:**
- `sparse_lu_factor(mat, pivot, tol)` — in-place LU decomposition
- `sparse_lu_factor_opts(mat, &opts)` — LU with optional fill-reducing reordering (RCM/AMD)
- `sparse_lu_solve(mat, b, x)` — solve using factored matrix (auto-unpermutes if reordered)
- `sparse_lu_condest(A, LU, &cond)` — estimate 1-norm condition number from LU factors
- `sparse_lu_refine(A, LU, b, x, max_iters, tol)` — iterative refinement

**CSR LU (high-performance path):**
- `lu_csr_from_sparse(A, fill_factor, &csr)` — convert to CSR working format
- `lu_csr_eliminate(csr, tol, drop_tol, piv)` — scatter-gather LU elimination
- `lu_csr_eliminate_block(csr, tol, drop_tol, min_block, piv)` — with dense block optimization
- `lu_csr_solve(csr, piv, b, x)` — forward/backward substitution in CSR
- `lu_csr_solve_block(csr, piv, B, nrhs, X)` — block solve for multiple RHS
- `lu_csr_factor_solve(A, b, x, tol)` — one-shot convert + factor + solve
- `lu_detect_dense_blocks(csr, min_size, threshold, &blocks, &nblocks)` — supernodal dense block detection

**Cholesky (SPD matrices):**
- `sparse_cholesky_factor(mat)` — in-place A = L·L^T
- `sparse_cholesky_factor_opts(mat, &opts)` — with optional AMD/RCM reordering
- `sparse_cholesky_solve(mat, b, x)` — solve using Cholesky factors

**LDL^T (symmetric indefinite matrices):**
- `sparse_ldlt_factor(A, &ldlt)` — P·A·P^T = L·D·L^T with Bunch-Kaufman 1x1/2x2 pivoting
- `sparse_ldlt_factor_opts(A, &opts, &ldlt)` — with optional AMD/RCM fill-reducing reordering
- `sparse_ldlt_solve(&ldlt, b, x)` — solve using LDL^T factors (auto-unpermutes)
- `sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero)` — eigenvalue sign count from D blocks
- `sparse_ldlt_refine(A, &ldlt, b, x, max_iters, tol)` — iterative refinement
- `sparse_ldlt_condest(A, &ldlt, &cond)` — 1-norm condition estimate via Hager/Higham
- `sparse_ldlt_free(&ldlt)` — free factorization data

**Symmetric eigensolvers (Sprint 20):**
- `sparse_eigs_sym(A, k, &opts, &result)` — k extreme or near-sigma eigenpairs of symmetric A via Lanczos (growing-m outer loop) with full MGS reorthogonalization
- `opts.which` = `SPARSE_EIGS_LARGEST` / `_SMALLEST` / `_NEAREST_SIGMA`; the shift-invert mode composes with `sparse_ldlt_factor_opts` (Sprint 20 Days 4-6 AUTO dispatch)
- `opts.compute_vectors = 1` populates `result.eigenvectors` (column-major, caller-owned); `result.used_csc_path_ldlt` reports the inner LDL^T backend for shift-invert

**Symbolic analysis & refactorization:**
- `sparse_analyze(A, &opts, &analysis)` — compute elimination tree, column counts, symbolic structure
- `sparse_factor_numeric(A, &analysis, &factors)` — numeric-only factorization using precomputed analysis
- `sparse_refactor_numeric(A_new, &analysis, &factors)` — refactor with new values (same pattern)
- `sparse_factor_solve(&factors, &analysis, b, x)` — solve using factors with auto-permutation
- `sparse_analysis_free(&analysis)` / `sparse_factor_free(&factors)` — cleanup

**QR factorization (rectangular & rank-deficient):**
- `sparse_qr_factor(A, &qr)` — column-pivoted QR: A*P = Q*R
- `sparse_qr_factor_opts(A, &opts, &qr)` — with optional AMD column reordering
- `sparse_qr_solve(&qr, b, x, &residual)` — least-squares: min ||Ax - b||
- `sparse_qr_apply_q(&qr, transpose, x, y)` — apply Q or Q^T to a vector
- `sparse_qr_rank(&qr, tol)` — numerical rank estimation
- `sparse_qr_nullspace(&qr, tol, basis, &ndim)` — null-space basis extraction
- `sparse_qr_solve_minnorm(A, b, x, &opts)` — minimum 2-norm solution for underdetermined systems
- `sparse_qr_diag_r(&qr, diag)` — extract R diagonal for rank inspection
- `sparse_qr_rank_info(&qr, tol, &info)` — comprehensive rank diagnostics with condition estimate
- `sparse_qr_condest(&qr)` — quick condition estimate from R diagonal
- `sparse_qr_refine_minnorm(A, b, x, iters, &resid, &opts)` — iterative refinement for minimum-norm
- `sparse_qr_free(&qr)` — free QR factors
- `sparse_reorder_colamd(A, perm)` — COLAMD column ordering for unsymmetric/QR (handles rectangular)

**SVD:**
- `sparse_svd_compute(A, &opts, &svd)` — full SVD: A = U·Σ·V^T (singular values only or with vectors)
- `sparse_svd_partial(A, k, &opts, &svd)` — k largest singular values via Lanczos bidiagonalization
- `sparse_cond(A, &err)` — 2-norm condition number via SVD
- `sparse_svd_rank(A, tol, &rank)` — numerical rank estimation
- `sparse_pinv(A, tol, &pinv)` — Moore-Penrose pseudoinverse
- `sparse_svd_lowrank(A, k, &dense)` — best rank-k approximation (dense output)
- `sparse_svd_lowrank_sparse(A, k, drop_tol, &sparse)` — best rank-k approximation (sparse output)
- `sparse_svd_free(&svd)` — free SVD result

**Iterative solvers:**
- `sparse_solve_cg(A, b, x, &opts, precond, ctx, &result)` — Preconditioned Conjugate Gradient (SPD only)
- `sparse_solve_gmres(A, b, x, &opts, precond, ctx, &result)` — Restarted GMRES(k) with left/right preconditioning
- `sparse_cg_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block CG for multiple RHS
- `sparse_gmres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block GMRES for multiple RHS
- `sparse_solve_cg_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free CG
- `sparse_solve_gmres_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free GMRES
- `sparse_solve_minres(A, b, x, &opts, precond, ctx, &result)` — MINRES for symmetric (possibly indefinite) systems
- `sparse_minres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block MINRES for multiple RHS
- `sparse_solve_bicgstab(A, b, x, &opts, precond, ctx, &result)` — BiCGSTAB for general nonsymmetric systems
- `sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block BiCGSTAB for multiple RHS
- `sparse_solve_bicgstab_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free BiCGSTAB

**ILU(0) / ILUT preconditioners:**
- `sparse_ilu_factor(A, &ilu)` — ILU(0) factorization (no fill-in beyond A's pattern)
- `sparse_ilut_factor(A, &opts, &ilu)` — ILUT with threshold dropping and controlled fill-in
- `sparse_ilu_solve(&ilu, r, z)` — apply preconditioner: solve L*U*z = r
- `sparse_ilu_precond` / `sparse_ilut_precond` — callbacks compatible with `sparse_precond_fn`
- `sparse_ilu_free(&ilu)` — free ILU/ILUT factors

**IC(0) preconditioner (incomplete Cholesky):**
- `sparse_ic_factor(A, &ic)` — IC(0) factorization for SPD matrices (L*L^T ≈ A, no fill-in)
- `sparse_ic_solve(&ic, r, z)` — apply preconditioner: solve L*L^T*z = r
- `sparse_ic_precond` — callback compatible with `sparse_precond_fn`
- `sparse_ic_free(&ic)` — free IC(0) factors

**Fill-reducing reordering:**
- `sparse_reorder_rcm(A, perm)` — Reverse Cuthill-McKee ordering
- `sparse_reorder_amd(A, perm)` — Approximate Minimum Degree ordering
- `sparse_permute(A, row_perm, col_perm, &B)` — apply permutation
- `sparse_bandwidth(A)` — compute matrix bandwidth

**Matrix arithmetic:**
- `sparse_matmul(A, B, &C)` — sparse matrix-matrix multiply (Gustavson's algorithm)
- `sparse_scale(mat, alpha)` — in-place scalar multiplication
- `sparse_add(A, B, alpha, beta, &C)` — C = alpha*A + beta*B
- `sparse_add_inplace(A, B, alpha, beta)` — A = alpha*A + beta*B
- `sparse_norminf(mat, &norm)` — infinity norm (cached)

**I/O and format conversion:**
- `sparse_save_mm(mat, filename)` / `sparse_load_mm(&mat, filename)` — Matrix Market format
- `sparse_to_csr(mat, &csr)` / `sparse_from_csr(csr, &mat)` — CSR conversion
- `sparse_to_csc(mat, &csc)` / `sparse_from_csc(csc, &mat)` — CSC conversion
- `sparse_errno()` — retrieve system errno after I/O failure

All functions return `sparse_err_t` error codes (except accessors that return values directly). See `sparse_strerror()` for human-readable error messages.

## Performance Characteristics

| Matrix type | Pivoting | Factorization | Fill-in |
|-------------|----------|---------------|---------|
| Tridiagonal (n=5000) | Partial | 0.5 ms | 1.00x (zero fill-in) |
| Tridiagonal (n=5000) | Complete | 322 ms | ~1.7x |
| west0067 (67×67) | Partial | 0.5 ms | 3.2x |
| nos4 (100×100, sym) | Partial | 0.6 ms | 2.5x |
| fs_541_1 (541×541) | Partial | 5.2 ms | 1.7x |
| orsirr_1 (1030×1030) | Partial | 1,744 ms | 11.4x |

### CSR LU Speedup

The CSR working format eliminates linked-list pointer chasing during elimination, achieving significant speedup on large matrices:

| Matrix | Linked-list | CSR | Speedup |
|--------|------------|-----|---------|
| orsirr_1 (1030×1030) | 1.38 s | 0.11 s | **12x** |

### CSC Cholesky Speedup (Sprint 17 + Sprint 18)

The CSC working-format kernel for Cholesky uses contiguous column
storage with a dense scatter-gather workspace, eliminating linked-list
pointer chasing in the column sweep (`cmod` + `cdiv`).  Sprint 18
Days 6-10 added a **batched supernodal path** (external cmod + dense
Cholesky factor + dense triangular panel solve) on top of the scalar
kernel.  On SuiteSparse SPD matrices (3-repeat one-shot factor, AMD
reorder included on all paths):

| Matrix        |    n   |   nnz(A)  | Linked-list factor | CSC scalar | CSC supernodal | Speedup (scalar / sn) |
|---------------|-------:|----------:|-------------------:|-----------:|---------------:|----------------------:|
| nos4.mtx      |    100 |       594 |     0.46 ms |    0.42 ms |      0.38 ms | **1.09× / 1.22×** |
| bcsstk04.mtx  |    132 |     3,648 |     3.12 ms |    2.67 ms |      3.09 ms | **1.16× / 1.01×** |
| bcsstk14.mtx  |  1,806 |    63,454 |   364.29 ms |  208.82 ms |    152.83 ms | **1.74× / 2.38×** |
| s3rmt3m3.mtx  |  5,357 |   207,123 |  4018.41 ms | 1914.53 ms |   1179.41 ms | **2.10× / 3.41×** |
| Kuu.mtx       |  7,102 |   340,200 |  3147.78 ms | 4112.76 ms |   1416.64 ms |   0.77× / **2.22×** |
| Pres_Poisson  | 14,822 |   715,804 | 46003.69 ms |17597.98 ms |  10580.68 ms | **2.61× / 4.35×** |

Residuals `||A·x − b||_∞ / ||b||_∞` match the linked-list path to
within double-precision round-off (≤ 2e-13) on every matrix above.
Numbers are 3-repeat averages measured with
`./build/bench_chol_csc --repeat 3`; full details in
[`docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`](docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md)
and the raw Day 12 capture in
[`docs/planning/EPIC_2/SPRINT_18/bench_day12.txt`](docs/planning/EPIC_2/SPRINT_18/bench_day12.txt).

The scalar-CSC speedup climbs from 1.09× at n = 100 to 2.61× at
n = 14 822 — consistent with linked-list pointer-chasing overhead
growing faster than contiguous column traversal.  The supernodal
path adds another 1.2–2.9× on top of scalar on every non-trivial
matrix (exception: bcsstk04, where supernode-detection overhead
eats the batched dense-block win).  Kuu's scalar regression (0.77×)
is localised to the `shift_columns_right_of` packing cost in drop-
tolerance pruning; the supernodal path pre-allocates the full
sym_L pattern and sidesteps the shifts, landing 2.22× ahead.

The table above is the **one-shot** case: AMD reordering runs on
every factor call on all paths.  In the analyze-once / factor-many
workflow (`sparse_analyze` + `sparse_factor_numeric`, Sprint 14) the
AMD cost is amortized across many numeric refactorizations with the
same pattern, and the CSC kernel's speedup over the linked-list
kernel is larger because only the numeric factor time remains in
the comparison.

**Transparent dispatch (Sprint 18 Day 11).**
`sparse_cholesky_factor_opts(mat, opts)` now routes through the CSC
supernodal kernel whenever `mat->rows >= SPARSE_CSC_THRESHOLD`
(default `100` in `include/sparse_matrix.h`), writing the factor
back into `mat` via `chol_csc_writeback_to_sparse`.  Callers do not
need to select a backend — the numbers above are what the public
entry point delivers.  `sparse_cholesky_opts_t::backend`
(`SPARSE_CHOL_BACKEND_AUTO` / `LINKED_LIST` / `CSC`) forces a path
for tests; `used_csc_path` reports which branch ran.  Smaller
matrices may see a slight slowdown from CSC conversion cost and are
left on the linked-list path.

### CSC LDL^T (Sprint 17 scaffolding + Sprint 18 native + Sprint 19 row-adj + supernodal)

The CSC LDL^T path (`ldlt_csc_factor` + `ldlt_csc_solve`) was a
wrapper in Sprint 17, replaced by a native column-by-column
Bunch-Kaufman kernel in Sprint 18 (1×1 / 2×2 pivot blocks, α = (1 +
√17) / 8 partial scan, symmetric swaps in packed CSC storage).
Sprint 19 added a per-row adjacency index (`row_adj`) so the cmod
inner loop iterates only contributing priors instead of `[0, step_k)`,
plus a batched supernodal kernel (`ldlt_csc_eliminate_supernodal`)
mirroring the Sprint 18 Cholesky batched path.  The **LL factor** and
**CSC native** columns below run under the one-shot fair-comparison
methodology (AMD inside the timed region on both sides).  The **CSC
supernodal** column is measured by `bench_ldlt_csc --supernodal`,
which uses an analyze-once / pre-permuted pipeline: a scalar pre-pass
resolves the BK permutation + pivot_size once up front, and each timed
repetition reuses those cached decisions and measures only the
pre-permuted conversion + supernodal factor.  The supernodal speedup
is therefore a steady-state analyze-once / factor-many number — it is
not directly comparable to the LL / CSC native one-shot columns (and
is correspondingly higher than a like-for-like one-shot comparison
would show).

| Matrix       |    n  |  nnz(A)  | LL factor  | CSC native | CSC supernodal (analyze-once) | Speedup (native one-shot / supernodal analyze-once) |
|--------------|------:|---------:|-----------:|-----------:|------------------------------:|----------------------------------------------------:|
| nos4.mtx     |   100 |      594 |    0.38 ms |    0.29 ms |                       0.14 ms |                                1.29× / **2.62×**    |
| bcsstk04.mtx |   132 |    3,648 |    3.76 ms |    2.16 ms |                       1.23 ms |                            **1.74×** / **3.05×**    |
| bcsstk14.mtx | 1,806 |   63,454 |  493.74 ms |  140.59 ms |                      72.29 ms |                            **3.51×** / **6.83×**    |

The Sprint 19 Day 9 row-adjacency index improved the native scalar
LDL^T kernel from Sprint 18's 2.45× on bcsstk14 to 3.51× by removing
the per-step prior-column scan from the cmod inner loop.  The
batched supernodal LDL^T (`--supernodal` mode) lifts that further to
6.83× on bcsstk14 by delegating supernode diagonal blocks to a
dense LDL^T primitive and solving panel rows en masse.  Residuals
match across paths to round-off.  The supernodal LDL^T path is
currently SPD-biased (heuristic CSC fill via `ldlt_csc_from_sparse`
covers SPD cmod fill but not always indefinite cmod fill — KKT-style
saddle points should fall back to scalar `ldlt_csc_eliminate`); a
`_with_analysis` mirror that pre-allocates full `sym_L` is the
natural Sprint 20 follow-up.

End-of-sprint snapshot in `docs/planning/EPIC_2/SPRINT_19/bench_day14.txt`
covers all three benchmarks (`bench_chol_csc`, `bench_ldlt_csc`, and
the new `bench_refactor_csc` analyze-once / factor-many harness)
with detailed Sprint 18 → Sprint 19 deltas.

**Complexity:**
- Partial pivoting: O(nnz) per elimination step — strongly preferred for banded/structured matrices
- Complete pivoting: O(n²) per elimination step due to submatrix search — better numerical stability but much slower
- Solve: O(nnz_LU) for forward/backward substitution
- SpMV: O(nnz)
- Block SpMV: O(nnz × nrhs) with improved cache locality

## Thread Safety

The library is safe for concurrent use under the following contract:

| Operation | Thread-safe? | Notes |
|-----------|:---:|-------|
| Concurrent solves on the same factored matrix | Yes | Solve reads `factor_norm` and linked-list structure (immutable after factorization) |
| Concurrent `sparse_norminf()` on the same matrix | Yes | `cached_norm` is `_Atomic double` with relaxed ordering; idempotent computation |
| Concurrent factorization of different matrices | Yes | Each matrix has its own pool allocator |
| Concurrent read-only access (nnz, get, matvec) | Yes | No shared mutable state |
| `sparse_errno()` | Yes | Uses `_Thread_local` storage |
| Concurrent mutation of the same matrix | **No** | Insert/remove/factor on a shared matrix requires external synchronization |
| Factorization concurrent with solve on same matrix | **No** | Factorization mutates structure; solve must wait until factorization completes |

**Mutable fields in SparseMatrix:**

| Field | Mutated by | Thread safety |
|-------|-----------|---------------|
| `cached_norm` | `sparse_norminf()` | `_Atomic double` — safe for concurrent reads/writes |
| `factor_norm` | Factorization functions | Written once during factorization, read during solve — no race (factorization completes before solve) |
| Pool, `row_headers`, `col_headers`, `nnz` | `sparse_insert()`, `sparse_remove()`, factorization | Not atomic — requires external synchronization or `SPARSE_MUTEX` |
| Permutation arrays | Factorization functions | Single-threaded context only |

**Optional mutex support:** Compile with `-DSPARSE_MUTEX` and `-pthread` to add per-matrix mutex locking on `sparse_insert()` and `sparse_remove()`. This serializes concurrent insert/remove calls on the same matrix. Note: factorization (`sparse_lu_factor`, `sparse_cholesky_factor`) is not mutex-protected and must not be called concurrently on the same matrix. Not recommended — prefer separate matrices per thread.

## Known Limitations

- **32-bit indices.** `idx_t` is `int32_t`, limiting matrix dimensions and nonzero counts to ~2.1 billion. To support larger matrices, change the typedef in `sparse_types.h` to `int64_t` and recompile.
- **In-place factorization.** `sparse_lu_factor` and `sparse_cholesky_factor` overwrite the matrix; always work on a copy if you need the original. (The CSR path via `lu_csr_factor_solve` does not modify the input.)
- **Factored-state validation.** Solve functions check an internal `factored` flag and return `SPARSE_ERR_BADARG` if the matrix has not been factored. Modifying a factored matrix (insert/remove) clears the flag. For externally-constructed factors (e.g., imported from CSR), call `sparse_mark_factored()` before solving.
- **No complex or integer matrices.** Only real (double-precision) values are supported.

## Testing

The test suite contains **1453 unit tests** across 42 test suites with >=95% line coverage (CI-enforced):

- Sparse matrix data structure, norms, symmetry, transpose (53 tests)
- LU factorization, solve, condition estimation (37 tests)
- Matrix Market I/O with errno validation (22 tests)
- Known reference matrices (15 tests)
- Vector utilities, SpMV, iterative refinement (24 tests)
- Edge cases, tolerance hardening, and factored-state validation (54 tests)
- Integration tests (7 tests)
- Matrix arithmetic — scale and add (23 tests)
- SuiteSparse real-world matrix validation (10 tests)
- Reordering — RCM, AMD, permutation (38 tests)
- Cholesky factorization and solve (21 tests)
- CSR/CSC conversion (11 tests)
- Sparse matrix-matrix multiply (14 tests)
- Thread safety (8 tests)
- Sprint 4 cross-feature integration (5 tests)
- Iterative solvers — CG, GMRES, matrix-free, SuiteSparse (76 tests)
- ILU(0) and ILUT preconditioners (34 tests)
- Parallel SpMV (12 tests)
- Sprint 5 cross-feature integration (14 tests)
- Sparse QR — Householder, least-squares, rank, null space, economy, sparse-mode (71 tests)
- Sprint 6 cross-feature integration (7 tests)
- Dense utilities — Givens, eigensolvers, tridiag QR (34 tests)
- Bidiagonal reduction (12 tests)
- SVD — full, partial, rank-deficient, condition number, pseudoinverse, low-rank (91 tests)
- Sprint 8 cross-feature integration (7 tests)
- Fuzz and property-based tests (24 tests)
- CSR LU — conversion, elimination, dense blocks, block solve, coverage gaps (53 tests)
- Block solvers — block SpMV, block CG, block GMRES (15 tests)
- Sprint 10 cross-feature integration (14 tests)
- Sprint 11 tolerance, factored-state, and version integration (6 tests)
- LDL^T factorization — Bunch-Kaufman pivoting, 2x2 blocks, reordering, KKT systems (72 tests)
- Sprint 12 LDL^T cross-feature integration (8 tests)
- IC(0) incomplete Cholesky — factor, solve, CG preconditioning, SuiteSparse (27 tests)
- MINRES solver — SPD, indefinite, preconditioned, block, robustness (43 tests)
- Sprint 13 IC(0) + MINRES cross-feature integration (14 tests)
- CSC Cholesky — alloc/convert/eliminate/solve, symbolic path, supernode detection, dense primitives (100 tests — Sprint 17)
- CSC LDL^T — alloc/convert/eliminate/solve, Bunch-Kaufman 1×1/2×2, linked-list cross-check, inertia (40 tests — Sprint 17)

```bash
make test          # run all tests
make smoke         # quick smoke test
make sanitize      # UBSan (undefined behavior)
make asan          # ASan (address sanitizer) — requires GCC or LLVM clang on macOS
make sanitize-all  # both ASan + UBSan
make tsan          # TSan (thread sanitizer) for concurrent tests
make coverage      # line coverage report (requires gcc + lcov + bc); fails if < 95%
```

**Note:** Apple Clang's ASan hangs on macOS. Use an alternative compiler:
```bash
CC=gcc-14 make asan
CC=/opt/homebrew/opt/llvm/bin/clang make asan
```
On Linux, `make asan` works with the default compiler.

## Project Structure

```
linalg_sparse_orthogonal/
├── include/              Public headers (16 headers)
│   ├── sparse_types.h        Error codes, index type (includes sparse_version.h)
│   ├── sparse_version.h      Version macros (generated from VERSION file)
│   ├── sparse_matrix.h       Core data structure, SpMV, block SpMV, I/O
│   ├── sparse_lu.h           LU factorization, solve, block solve
│   ├── sparse_lu_csr.h       CSR LU — scatter-gather elimination, dense blocks
│   ├── sparse_cholesky.h     Cholesky factorization and solve
│   ├── sparse_iterative.h    CG, GMRES, MINRES, BiCGSTAB; block variants; GMRES left/right precond
│   ├── sparse_ilu.h          ILU(0) and ILUT preconditioners
│   ├── sparse_ic.h           IC(0) incomplete Cholesky preconditioner
│   ├── sparse_qr.h           QR factorization, least-squares, rank, null space
│   ├── sparse_dense.h        Dense utilities, Givens, eigensolvers
│   ├── sparse_bidiag.h       Householder bidiagonalization
│   ├── sparse_csr.h          CSR/CSC conversion
│   ├── sparse_reorder.h      Fill-reducing reordering (RCM, AMD, COLAMD)
│   ├── sparse_svd.h          SVD, condition number, pseudoinverse, low-rank
│   └── sparse_vector.h       Dense vector utilities
├── src/                  Library implementation (15 source files, ~10K lines)
├── tests/                Unit tests (35 suites, 976 tests, ~30K lines)
├── cmake/                CMake config templates
├── examples/             Standalone example programs + CMake integration example
├── benchmarks/           Performance benchmarks (5 programs)
├── docs/                 Algorithm/format documentation + planning
│   └── planning/EPIC_1/  Sprint plans, retrospectives, and project plan
├── INSTALL.md            Cross-platform installation guide
├── sparse.pc.in          pkg-config template
└── archive/              Original prototype files
```

## Installation

See [INSTALL.md](INSTALL.md) for detailed instructions covering Linux, macOS, and Windows. Quick summary:

```bash
# Makefile
make && make test && make install PREFIX=/usr/local

# CMake
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local && cmake --build build && cmake --install build
```

After installation, downstream projects can use:
- **pkg-config:** `pkg-config --cflags --libs sparse`
- **CMake:** `find_package(Sparse REQUIRED)` + `target_link_libraries(... Sparse::sparse_lu_ortho)`

## Documentation

- [Algorithm Description](docs/algorithm.md) — data structure, LU algorithm, complexity analysis
- [Matrix Market Format](docs/matrix_market.md) — supported features and limitations
- [Installation Guide](INSTALL.md) — cross-platform build and install instructions

## License

This project is for research and educational purposes.
