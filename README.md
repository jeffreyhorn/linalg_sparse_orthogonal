# linalg_sparse_orthogonal

A C library for sparse matrices using the **orthogonal linked-list** (cross-linked) representation, with direct and iterative linear system solvers.

## Start Here

Use the README front door for the first adoption path, then widen into the
deeper support surfaces only when you actually need them.

- **Want a first successful solve quickly?**
  - Build locally with `make`, then use the one-shot direct quick start below.
- **Need to choose the right solver workflow first?**
  - Jump to [Choose a Workflow](#choose-a-workflow).
- **Need install or downstream-consumer setup?**
  - Use [Installation](#installation) for the compact package summary, then
    [INSTALL.md](INSTALL.md) for platform-specific detail.
- **Need maintained examples, benchmarks, or quality policy?**
  - Examples stay in `examples/`, performance/reporting stays under
    `benchmarks/`, and maintainer/quality policy stays in
    [docs/maintainer_guide.md](docs/maintainer_guide.md).

## Features

### Core Data Structure
- **Orthogonal linked-list storage** — each non-zero is linked into both its row list and column list, enabling efficient row and column traversal
- **Slab pool allocator** with free-list for fast node allocation and reuse

### Direct Solvers
- **One-shot direct solves** — LU, Cholesky, LDL^T, and QR remain the default public entry points for most callers.
- **Explicit repeated-run direct lifecycle** — `sparse_analyze()` → `sparse_factor_numeric()` → `sparse_factor_solve()`, then `sparse_refactor_numeric()` between later `sparse_factor_solve()` calls, supports analyze-once / factor-many workflows when the sparsity pattern stays fixed.
- **Dispatch-backed direct kernels** — CSR LU plus CSC Cholesky and LDL^T provide faster large-matrix paths behind the existing public APIs.
- **Multi-RHS and refinement support** — block solves, iterative refinement, rank diagnostics, and minimum-norm QR paths stay available without changing the one-shot-first workflow.

### Singular Value Decomposition (SVD)
- **Full and partial SVD** — dense full SVD plus Lanczos-based partial SVD for the largest singular values.
- **Conditioning and inverse-style tools** — numerical rank, condition number estimation, pseudoinverse, and low-rank approximation.
- **Low-rank output choices** — dense and sparse low-rank approximations stay available under the same public SVD surface.

### Iterative Solvers
- **Core one-shot solvers** — CG for SPD systems, GMRES for general unsymmetric systems, MINRES for symmetric indefinite systems, and BiCGSTAB as a one-shot compatibility path.
- **Repeated-run iterative handles** — explicit reusable-handle support is intentionally bounded to `CG`, `GMRES`, and `MINRES`.
- **Preconditioning and matrix-free variants** — ILU(0), ILUT, IC(0), and matrix-free callbacks stay available without changing the public workflow boundary.
- **Diagnostics** — residual histories, stagnation detection, and breakdown reporting remain built into the iterative solver surfaces.

### Eigenvalue Infrastructure
- **Symmetric tridiagonal QR algorithm** — implicit QR with Wilkinson shifts and deflation (eigenvalues via `tridiag_qr_eigenvalues`, eigenpairs via `tridiag_qr_eigenpairs`)
- **2×2 symmetric eigensolver** — numerically stable quadratic formula
- **Dense matrix utilities** — Givens rotations, matrix-matrix/vector multiply

### Sparse Symmetric Eigensolver
- **`sparse_eigs_sym`** — k extreme or near-sigma eigenpairs of a symmetric sparse matrix. Three concrete backends are available behind `opts->backend = SPARSE_EIGS_BACKEND_AUTO` (overridable explicitly):
  - **Grow-m Lanczos** (`SPARSE_EIGS_BACKEND_LANCZOS`) — full MGS reorthogonalization with a growing-subspace outer loop. Peak memory `O(m_cap · n)`. AUTO default for `n < SPARSE_EIGS_THICK_RESTART_THRESHOLD` (500).
  - **Wu/Simon thick-restart Lanczos** (`SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`) — preserves the converged Ritz subspace in a compact arrowhead basis between restart phases; peak memory `O((k + m_restart) · n)` regardless of total iteration count. AUTO default for `n ≥ 500` when no preconditioner is supplied.
  - **LOBPCG** (`SPARSE_EIGS_BACKEND_LOBPCG`) — Knyazev's Locally Optimal Block Preconditioned Conjugate Gradient with block Rayleigh-Ritz over the `[X | W | P]` subspace, BLOPEX-style conditioning guard, and per-column soft-locking. It composes with the existing IC(0) and LDL^T preconditioner callbacks and AUTO routes here for `n ≥ SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` (1000) when `opts->precond != NULL` and the block size is at least 4.
- Three `which` modes across all backends — `LARGEST`, `SMALLEST`, `NEAREST_SIGMA`. Shift-invert composes with `sparse_ldlt_factor_opts` and the existing LDL^T backend dispatch.
- **Parallel MGS reorthogonalization** — both Lanczos backends parallelise the inner-product / daxpy bodies under `-DSPARSE_OPENMP`, gated on `n ≥ SPARSE_EIGS_OMP_REORTH_MIN_N` (500) so small problems don't pay OMP fork/join overhead.
- **Ritz-pair output** — optional eigenvectors via `compute_vectors = 1`; `result.residual_norm` reports the maximum relative Ritz residual across the converged pairs.
- **Observability** — `result.used_csc_path_ldlt` reports the LDL^T backend chosen on the shift-invert path; `result.peak_basis_size` reports the simultaneously-live `V` columns for memory budgeting; `result.backend_used` reports which backend AUTO actually picked.
- **Optional inverse-iteration refinement post-pass** — `opts->refine = 1` runs Rayleigh-quotient inverse iteration on each converged Ritz pair, refactoring `(A − λ_j I) = L D L^T` per-pair via `sparse_ldlt_factor_opts` (with retry under a small perturbation if the shifted matrix is singular at the converged Ritz value). Default off; budget-bound via `opts->refine_max_iters` (default 5).
- **Preconditioning speedup** — on bcsstk04 (n = 132, cond ≈ 5e6) k = 3 SMALLEST: vanilla LOBPCG saturates the 800-iteration cap with residual ~1e+01, IC(0) preconditioning converges in **62 iterations** at residual 8e-9, LDL^T preconditioning converges in **8 iterations** at residual 3e-9.
- **`bench_eigs`** — permanent benchmark driver at `benchmarks/bench_eigs.c`; CLI with `--sweep default`, `--compare`, and `--matrix <path>` modes, CSV output, and configurable `--repeats`. `benchmarks/README.md` documents the current CLI and CSV schema.

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
- **Fill-reducing reordering** — Reverse Cuthill-McKee (RCM); Approximate Minimum Degree (AMD, ~5·nnz + 6·n + 1 initial integer workspace, growing on demand when fill-in pushes adjacency past the initial bound — scales to large structurally regular fixtures without the earlier bitset's O(n²/64) penalty); Nested Dissection (ND, multilevel vertex separator — best on 2D / 3D PDE meshes); and Column Approximate Minimum Degree (COLAMD) for unsymmetric/QR problems. On the explicit `sparse_analyze()` lifecycle, `sparse_analysis_opts_t.reorder_opts` now carries the shipped typed analysis-time controls for supernodal etree postorder and the highest-value ND routing/coarsening knobs. Legacy `SPARSE_SUPERNODAL_POSTORDER` / `SPARSE_ND_*` env vars remain compatibility overrides only when those typed fields are left unspecified. `bench_reorder --reorder-via-analyze` exercises the analyze-time dispatch path directly from the bench harness.
- **Condition number estimation** — Hager/Higham 1-norm estimator from LU or LDL^T factors, quick R-diagonal estimator from QR

### I/O & Interop
- **Matrix Market I/O** — load and save `.mtx` files (coordinate real general, symmetric, and pattern formats)
- **CSR/CSC export plus compressed-first construction** — convert to/from compressed sparse row/column formats and enter the one-shot direct workflow directly from caller-owned compressed data

### Quality
- **Thread-safe** — concurrent solves on shared factored matrices, per-matrix pool allocators
- **Parallel SpMV** — OpenMP row-wise parallelization (compile with `-DSPARSE_OPENMP`)
- **errno capture** for I/O errors (`sparse_errno`)
- **Progress / cancel callbacks** (Sprint 29 Days 6-7) — `sparse_progress_cb_t` + `opts->progress_cb` / `opts->progress_user` across LU (linked-list + CSR), Cholesky (linked-list), LDL^T (linked-list), QR, CG, GMRES, MINRES, BiCGSTAB, grow-m Lanczos, and LOBPCG.  The CSC supernodal Cholesky / LDL^T kernels and the Wu/Simon thick-restart Lanczos outer loop are NOT wired (Sprint 30+ follow-up).  Callback signature emits `phase` / `step` / `total` / `elapsed_s`; a non-zero return cancels with `SPARSE_ERR_CANCELLED`.  **Cancellation semantics are family/path-local:** LU no-reorder one-shot cancellation at step 0 preserves the caller matrix, reordered LU one-shot attempts preserve the caller matrix through a temporary reordered working copy, Cholesky no-reorder linked-list cancellation is not bit-identical because the upper triangle is stripped before the first emission, reordered Cholesky one-shot attempts preserve the caller matrix through a temporary reordered working copy, and LDL^T / QR leave the input matrix bit-identical because factor state is separately owned.  Iterative solvers and eigensolvers don't write to `A` at all.  See `include/sparse_lu.h` / `include/sparse_cholesky.h` / `include/sparse_ldlt.h` opts headers for the per-routine contract.  Default `NULL` callback runs at zero overhead (no `make wall-check` regression vs Sprint 28).
- **Continuous integration** — Linux remains the strongest reviewed source of truth (`make quality-review-compile`, reviewed CMake parity, dead-code); macOS enforces the Apple Clang reviewed path with supplemental Homebrew GCC and static-first Make install/`pkg-config` verification; Windows enforces the reviewed CMake subset and the CMake-first consumer story. ThreadSanitizer stays on Linux (macOS-15+ TSan blocked by an upstream dyld issue), and `make bench-fast` remains the bounded PR-time runtime benchmark signal.

## Choose a Workflow

Start with the smallest surface that matches your real workload:

- **One-shot direct solve:** use LU, Cholesky, LDL^T, or QR when you are
  solving once or only occasionally.
- **Compressed-first one-shot direct entry:** use
  `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)` when your
  matrix already lives in compressed sparse storage and you want a one-shot
  direct path without treating linked-list mutation as the starting point.
- **Stable-pattern repeated direct solve:** use
  `sparse_analyze()` → `sparse_factor_numeric()` →
  `sparse_factor_solve()`, then `sparse_refactor_numeric()` between later
  solves as values change.
- **Repeated-run iterative handle:** use the explicit handle path only for
  `CG`, `GMRES`, or `MINRES` when the problem dimension is stable and
  workspace reuse matters.
- **Repeated-run eigensolver handle:** use the explicit handle path for
  grow-m Lanczos, thick-restart Lanczos, or explicit `LOBPCG` when you are
  reusing the same dimension and want persistent workspace.

Start from these shipped references:

- `example_basic_solve` for the smallest one-shot direct path
- `example_analysis` for the analyze-once / factor-many direct lifecycle
- [docs/tutorial.md](docs/tutorial.md) for the fuller repeated-run direct flow

When to widen beyond the first examples:

- examples teach the API workflow
- benchmarks prove the retained workflow/performance story
- tests own regression, oracle, and property guarantees
- `make bench-canonical-report` writes one bounded snapshot of the maintained
  benchmark surface and is intentionally not a pass/fail timing gate

If you still need the original coefficient view later, start one-shot direct
paths from a fresh matrix or a fresh `sparse_copy()`.

## Building

Most first-time local adoption only needs:

```bash
make
make test
```

Use `make tooling-build` when you want the example and benchmark binaries
without running them yet. Use [INSTALL.md](INSTALL.md) when you need
cross-platform install, downstream-consumer, or package-manager detail.

### With Make (recommended)

```bash
make            # build library
make tooling-build  # compile benchmark/example binaries without running them
make lint       # strict compile + static analysis (includes tooling-build)
make quality-review-compile  # reviewed format-check + lint wrapper
make test       # run all unit tests
make quality-review  # reviewed format-check + lint + test + deadcode-check
make quality-review-full  # strongest local reviewed baseline: quality-review + quality-review-cmake
make warning-workflow WARNING_WORKFLOW_LABEL=label  # authoritative repository-wide warning inventory capture
make quality-review-cmake-compile  # reviewed CMake configure + rebuild + ctest -N
make quality-review-cmake  # reviewed CMake configure + rebuild + ctest -N + ctest
make deadcode   # refresh raw dead-code evidence in build/deadcode/
make deadcode-report  # generate classified dead-code report.md / report.tsv
make deadcode-check   # verify report completeness invariants
make bench      # run benchmarks
make bench-canonical-report  # write one CSV per canonical maintained benchmark under build/bench-reports/canonical/
make examples   # build standalone example programs
make docs       # generate Doxygen API reference (requires doxygen)
make omp        # build and test with OpenMP-enabled parallel SpMV
make sanitize   # build with undefined-behavior sanitizer
make coverage   # default line-coverage report on the active test surface (80% threshold; backend auto-selected)
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

Use this path if you want one successful direct solve before learning the
repeated-run direct lifecycle or iterative/eigensolver surfaces.

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

Next steps after the first solve:

- stay on the one-shot path for small or occasional direct solves
- move to a compressed-first one-shot entry path when your matrix already
  arrives as CSR or CSC data
- move to [Repeated-Run Direct Workflow](#repeated-run-direct-workflow) when
  the sparsity pattern is stable across many value changes
- move to [Iterative Solver Example](#iterative-solver-example) when the
  matrix/system type makes iterative workflows a better fit
- use [Installation](#installation) and [INSTALL.md](INSTALL.md) when you need
  installed consumer workflows instead of local build-tree linking

If your coefficients already live in compressed sparse storage, the smallest
one-shot direct entry path is now:

- `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)` to build the
  public matrix shell from caller-owned compressed data
- then the usual one-shot direct family API on that matrix shell

That keeps the linked-list shell as the mutable compatibility owner, but it
avoids forcing compressed-input callers to think of incremental shell mutation
as their natural starting point.

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

    sparse_gmres_opts_t opts = {
        .max_iter = 1000,
        .restart = 50,
        .tol = 1e-10,
    };
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

Use IC(0) with SPD iterative workflows and ILU(0) / ILUT with general or
indefinite-system workflows. Preconditioner setup routines expect the original
matrix state with identity permutations, so if a matrix may already have been
factored or reordered, start from a fresh `sparse_copy()` of the original.

### Repeated-Run Lifecycle Handles

The library exposes an explicit repeated-run handle path for callers solving
many same-dimension problems while wanting to preserve allocation capacity
between runs.

The one-shot APIs remain fully supported:

- `sparse_solve_cg(...)`
- `sparse_solve_gmres(...)`
- `sparse_solve_minres(...)`
- `sparse_eigs_sym(...)`

Use them when:

- you are solving once or only occasionally
- simplicity matters more than workspace reuse
- you do not want to manage a prepare / free lifecycle explicitly

Use the explicit handle path when:

- the problem dimension is stable across repeated solves
- allocator churn or repeated workspace setup is worth avoiding
- you want one caller-owned object whose capacity can be prepared once and
  reused across runs

The public repeated-run iterative surface is:

- `sparse_iter_handle_t`
- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_iter_handle_prepare_minres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`
- `sparse_solve_minres_with_handle(...)`
- `sparse_iter_handle_free(...)`

The public repeated-run eigensolver surface is:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`
- `sparse_eigs_handle_free(...)`

The lifecycle contract is:

1. zero-initialize the handle or call the init helper
2. optionally prepare it for the stable dimension / working-set shape
3. run the corresponding `*_with_handle(...)` entry as many times as needed
4. free the handle when done

Important behavior:

- reusing a handle preserves allocation capacity, not old numerical iteration
  state
- re-preparing may grow capacity and discards prior Krylov / Ritz /
  search-direction state
- public repeated-run iterative handles are intentionally limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- the library does not expose public repeated-run handles for:
  - `BiCGSTAB`
  - block iterative workflows
- existing one-shot entries remain the compatibility path and are not
  deprecated

### Repeated-Run Direct Workflow

The direct-solver side uses a different public repeated-run shape from the
iterative and eigensolver handles. The explicit repeated-run direct path is:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`
- `sparse_analysis_free(...)`
- `sparse_factor_free(...)`

The intended lifecycle is:

1. zero-initialize `sparse_analysis_t` and `sparse_factors_t`
2. analyze once for the chosen direct family
3. factor / solve
4. refactor / solve many on same-pattern value changes
5. free both objects explicitly when done

Important behavior:

- the one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- the compressed-first one-shot entry path is:
  - build the public matrix shell from caller-owned CSR/CSC data with
    `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)`
  - then use the one-shot direct family API when you only need occasional
    solves
- repeated direct reuse preserves symbolic/permutation setup, not old numeric
  factor contents
- failed `sparse_refactor_numeric(...)` calls preserve the previously usable
  factor state on the public repeated-run direct path
- that same old-factor-preservation rule now holds on the large-`n`
  CSC-backed Cholesky lane as well; same-pattern non-SPD retries and obvious
  nnz drift reject instead of silently destroying the previous factor
- `sparse_refactor_numeric(...)` is the public same-pattern numeric-refresh
  path, not a general “accept any changed matrix” rebuild path
- the library now rejects obvious gross-structure drift cheaply, but it does
  not promise a full structural-pattern verifier

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
| [`sparse_iterative.h`](include/sparse_iterative.h) | CG, GMRES, MINRES, BiCGSTAB; block CG/GMRES/MINRES; GMRES left/right preconditioning; explicit repeated-run handles for CG/GMRES/MINRES |
| [`sparse_ilu.h`](include/sparse_ilu.h) | ILU(0) and ILUT incomplete factorization preconditioners |
| [`sparse_ic.h`](include/sparse_ic.h) | IC(0) incomplete Cholesky preconditioner for SPD systems |
| [`sparse_qr.h`](include/sparse_qr.h) | Column-pivoted QR factorization, least-squares, rank, null space, refinement |
| [`sparse_dense.h`](include/sparse_dense.h) | Dense matrix utilities, Givens rotations, 2×2 eigensolver, tridiag QR |
| [`sparse_bidiag.h`](include/sparse_bidiag.h) | Householder bidiagonalization (SVD preprocessing) |
| [`sparse_csr.h`](include/sparse_csr.h) | CSR/CSC compressed format conversion plus compressed-first matrix construction |
| [`sparse_reorder.h`](include/sparse_reorder.h) | Fill-reducing reordering (RCM, AMD, ND, COLAMD), permutation, bandwidth |
| [`sparse_svd.h`](include/sparse_svd.h) | SVD, partial SVD, condition number, pseudoinverse, low-rank approximation |
| [`sparse_eigs.h`](include/sparse_eigs.h) | Sparse symmetric eigensolver — Lanczos/LOBPCG backends, shift-invert mode, Ritz pairs, explicit repeated-run handle |
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
- `sparse_lu_factor_opts(mat, &opts)` — LU with optional fill-reducing reordering (RCM/AMD/ND)
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
- `sparse_cholesky_factor_opts(mat, &opts)` — with optional AMD/RCM/ND reordering
- `sparse_cholesky_solve(mat, b, x)` — solve using Cholesky factors

**LDL^T (symmetric indefinite matrices):**
- `sparse_ldlt_factor(A, &ldlt)` — P·A·P^T = L·D·L^T with Bunch-Kaufman 1x1/2x2 pivoting
- `sparse_ldlt_factor_opts(A, &opts, &ldlt)` — with optional AMD/RCM/ND fill-reducing reordering
- `sparse_ldlt_solve(&ldlt, b, x)` — solve using LDL^T factors (auto-unpermutes)
- `sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero)` — eigenvalue sign count from D blocks
- `sparse_ldlt_refine(A, &ldlt, b, x, max_iters, tol)` — iterative refinement
- `sparse_ldlt_condest(A, &ldlt, &cond)` — 1-norm condition estimate via Hager/Higham
- `sparse_ldlt_free(&ldlt)` — free factorization data

**Symmetric eigensolvers (Sprint 20):**
- `sparse_eigs_sym(A, k, &opts, &result)` — k extreme or near-sigma eigenpairs of symmetric A via Lanczos (growing-m outer loop) with full MGS reorthogonalization
- `sparse_eigs_handle_init(&handle)` / `sparse_eigs_handle_prepare(&handle, n, k, &opts)` / `sparse_eigs_sym_with_handle(A, k, &opts, &result, &handle)` / `sparse_eigs_handle_free(&handle)` — explicit repeated-run lifecycle path for stable-dimension symmetric eigensolves
- `opts.which` = `SPARSE_EIGS_LARGEST` / `_SMALLEST` / `_NEAREST_SIGMA`; the shift-invert mode composes with `sparse_ldlt_factor_opts` (Sprint 20 Days 4-6 AUTO dispatch)
- `opts.compute_vectors = 1` populates `result.eigenvectors` (column-major, caller-owned); `result.used_csc_path_ldlt` reports the inner LDL^T backend for shift-invert

**Symbolic analysis & refactorization:**
- `sparse_analyze(A, &opts, &analysis)` — compute elimination tree, column counts, symbolic structure
- `sparse_factor_numeric(A, &analysis, &factors)` — numeric-only factorization using precomputed analysis
- `sparse_refactor_numeric(A_new, &analysis, &factors)` — refactor with new values (same pattern)
- `sparse_factor_solve(&factors, &analysis, b, x)` — solve using factors with auto-permutation
- `sparse_analysis_free(&analysis)` / `sparse_factor_free(&factors)` — cleanup

The direct repeated-run contract is therefore:

- analyze once
- factor / solve
- refactor / solve many

with reuse preserving symbolic/permutation setup rather than stale numeric
factor contents.

**QR factorization (rectangular & rank-deficient):**
- `sparse_qr_factor(A, &qr)` — column-pivoted QR: A*P = Q*R
- `sparse_qr_factor_opts(A, &opts, &qr)` — with optional AMD column reordering
- `sparse_qr_solve(&qr, b, x, &residual)` — least-squares for overdetermined systems; basic solution for underdetermined systems
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
- `sparse_iter_handle_init(&handle)` / `sparse_iter_handle_prepare_cg(&handle, n)` / `sparse_solve_cg_with_handle(A, b, x, &opts, precond, ctx, &result, &handle)` / `sparse_iter_handle_free(&handle)` — explicit repeated-run CG lifecycle path
- `sparse_solve_gmres(A, b, x, &opts, precond, ctx, &result)` — Restarted GMRES(k) with left/right preconditioning
- `sparse_iter_handle_prepare_gmres(&handle, n, restart)` / `sparse_solve_gmres_with_handle(A, b, x, &opts, precond, ctx, &result, &handle)` — explicit repeated-run GMRES lifecycle path
- `sparse_iter_handle_prepare_minres(&handle, n)` / `sparse_solve_minres_with_handle(A, b, x, &opts, precond, ctx, &result, &handle)` — explicit repeated-run MINRES lifecycle path
- `sparse_cg_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block CG for multiple RHS
- `sparse_gmres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block GMRES for multiple RHS
- `sparse_solve_cg_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free CG
- `sparse_solve_gmres_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free GMRES
- `sparse_solve_minres(A, b, x, &opts, precond, ctx, &result)` — MINRES for symmetric (possibly indefinite) systems
- `sparse_minres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block MINRES for multiple RHS
- `sparse_solve_bicgstab(A, b, x, &opts, precond, ctx, &result)` — BiCGSTAB for general nonsymmetric systems
- `sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block BiCGSTAB for multiple RHS
- `sparse_solve_bicgstab_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free BiCGSTAB

The public repeated-run iterative handle support remains intentionally bounded
to `CG`, `GMRES`, and `MINRES`; `BiCGSTAB` and block iterative workflows
remain one-shot compatibility surfaces.

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
- `sparse_reorder_nd(A, perm)` — Nested Dissection (multilevel vertex-separator); best on 2D / 3D PDE meshes
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
- `sparse_to_csr(mat, &csr)` / `sparse_create_from_csr(csr)` — CSR export and compressed-first construction; free exported storage with `sparse_csr_free(csr)`
- `sparse_to_csc(mat, &csc)` / `sparse_create_from_csc(csc)` — CSC export and compressed-first construction; free exported storage with `sparse_csc_free(csc)`
- `sparse_from_csr(csr, &mat)` / `sparse_from_csc(csc, &mat)` — retained compatibility wrappers when you need explicit `sparse_err_t` import status
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

That repeated-run CSC story stays intentionally simple on the Cholesky side:

- AUTO picks linked-list vs CSC by size
- forcing CSC means the CSC backend directly
- the highest-signal repeated-run proof surfaces are:
  - `bench_refactor`
  - default SPD mode in `bench_refactor_csc`
- the family-local large-`n` analysis-backed CSC helper route stays owned by:
  - `tests/test_chol_csc.c`
- the public one-shot vs explicit repeated-run parity/error-path contract stays
  owned by:
  - `tests/test_integration.c`
  - including the large-`n` same-pattern LDL^T lifecycle oracle that now
    mirrors the one-shot CSC-backed LDL^T lane
- the bounded seeded generative follow-through for the same large-`n`
  CSC-backed lifecycle lane stays owned by:
  - `tests/test_fuzz.c`
  - including the large-`n` LDL^T CSC lifecycle property lane
- examples and benchmark surfaces stay intentionally outside that regression
  ownership split:
  - `example_analysis` teaches the repeated-run workflow
  - `bench_refactor` / `bench_refactor_csc` prove retained workflow and
    performance behavior
  - they do not replace the test-owned oracle/property lanes above

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
left on the linked-list path.  The maintained benchmark proof surface
`bench_chol_csc` now also reports `csc_scalar_path`,
`csc_supernodal_path`, `csc_supernodal_dense_kernel`, and
`csc_supernodal_panel_solver`; on the
default build those identify the current Sprint 64 backend-aware
supernodal lane as `scalar`, `supernodal`, `builtin`, and
`batched_panel` respectively.  If that internal dense-kernel descriptor or one
of its required callbacks cannot be resolved on the supernodal lane, the
public error taxonomy now reports `SPARSE_ERR_BACKEND_CONTRACT`
instead of collapsing that impossible internal seam into
`SPARSE_ERR_BADARG`.

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
match across paths to round-off.

Sprint 53 tightened the public LDL^T CSC interpretation beyond that
historical Sprint 19 snapshot:

- `sparse_ldlt_factor_opts(...)`
  - still gives callers the same one-shot AUTO / forced-backend interface
- forcing CSC now means the CSC **pipeline**, not a blanket promise that the
  batched completion path wins every indefinite input
- the scalar Bunch-Kaufman pre-pass remains the authoritative indefinite
  permutation-resolution step
- once that CSC pipeline is selected, completion may:
  - retain the batched path
  - or fall back to the resolved scalar-prepass factor when the batched path
    rejects the cached pivot pattern

That layering is intentionally different from Cholesky's simpler CSC story.
Both families have size-based AUTO dispatch, but LDL^T keeps the extra
indefinite permutation-resolution layer because symmetric indefinite CSC
completion is not just "Cholesky with a different dense kernel."

Sprint 53 also added a bounded indefinite repeated-run proof surface:

- `bench_refactor_csc --indefinite-kkt`
  - measures the public repeated-run LDL^T path against the direct
    resolved-analysis CSC completion path on a same-pattern KKT workload
  - closes at round-off residuals on both sides after the Sprint 53
    permutation-contract fix

So the current compact public interpretation is:

- Cholesky CSC dispatch
  - simpler size-based linked-list vs CSC selection
- LDL^T CSC dispatch
  - size-based outer selection plus the scalar BK pre-pass and CSC-pipeline
    completion rules above
- repeated-run benchmark proof
  - benchmark-local source of truth lives in `benchmarks/README.md`
  - default `bench_refactor_csc` mode covers SPD / Cholesky
  - `--indefinite-kkt` covers LDL^T on the bounded same-pattern KKT workload
  - benchmark proof stays distinct from the test-owned LDL^T oracle/property
    lanes in `tests/test_integration.c` and `tests/test_fuzz.c`

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

- **Default reviewed width remains 32-bit.** The shipped reviewed build uses
  `SPARSE_IDX_BITS=32`, so `idx_t` still limits matrix dimensions and nonzero
  counts to ~2.1 billion. Wider indices are now a bounded compile-time seam
  through `SPARSE_IDX_BITS=64`; downstream callers must rebuild against that
  same width contract.
- **In-place factorization.** `sparse_lu_factor` and `sparse_cholesky_factor` overwrite the matrix; always work on a copy if you need the original. (The CSR path via `lu_csr_factor_solve` does not modify the input.)
- **Factored-state validation.** Solve functions check an internal `factored` flag and return `SPARSE_ERR_BADARG` if the matrix has not been factored. Modifying a factored matrix (insert/remove) clears the flag. For externally-constructed factors (e.g., imported from CSR), call `sparse_mark_factored()` before solving.
- **Scalar support is still real-only.** The current dense-scalar public seam
  is named `sparse_scalar_t`, but the shipped contract still binds it to real
  double precision only. This is bounded preparation for later widening, not a
  claim of complex or broad generic-scalar support today.

## Testing

The maintained default regression surface currently registers **53** test
binaries in CTest. Coverage is a separate supplemental signal: the Linux
coverage job enforces an **80%** line-coverage threshold on `src/` for the
default instrumented test run, not for every opt-in test path automatically.

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
- Fuzz and property-based tests (25 tests)
- CSR LU — conversion, elimination, dense blocks, block solve, coverage gaps (53 tests)
- Block solvers — block SpMV, block CG, block GMRES (15 tests)
- Sprint 10 cross-feature integration (14 tests)
- Sprint 11 tolerance, factored-state, and version integration (6 tests)
- LDL^T factorization — Bunch-Kaufman pivoting, 2x2 blocks, reordering, KKT systems (72 tests)
- Sprint 12 LDL^T cross-feature integration (8 tests)
- IC(0) incomplete Cholesky — factor, solve, CG preconditioning, SuiteSparse (27 tests)
- MINRES solver — SPD, indefinite, preconditioned, block, robustness (43 tests)
- Sprint 13 IC(0) + MINRES cross-feature integration (14 tests)
- CSC Cholesky — alloc/convert/eliminate/solve, symbolic path, supernode detection, dense primitives, analysis-backed CSC parity (145 tests)
- CSC LDL^T — alloc/convert/eliminate/solve, Bunch-Kaufman 1×1/2×2, linked-list cross-check, inertia, supernodal follow-through (96 tests)

```bash
make test          # run all tests
make smoke         # quick smoke test
make sanitize      # UBSan (undefined behavior)
make asan          # ASan (address sanitizer) — requires GCC or LLVM clang on macOS
make sanitize-all  # both ASan + UBSan
make tsan          # TSan (thread sanitizer) for concurrent tests
make coverage      # line coverage report for the default active test surface; fails if < 80%
```

### Test Category Policy

The default regression surface is the set of tests registered with plain
`RUN_TEST(...)` in each `tests/test_*.c` binary. Those tests must pass under
ordinary `make test` and `ctest`.

The test framework also supports two explicit opt-in categories for live
non-default checks:

```bash
SPARSE_TEST_SLOW=1 make test
SPARSE_TEST_EXPERIMENTAL=1 make test
```

- `RUN_TEST_SLOW(...)` is for current supported behavior whose only default-path
  problem is runtime or fixture cost.
- `RUN_TEST_EXPERIMENTAL(...)` is for current live behavior on intentionally
  non-default paths that still must pass when enabled.
- Some suite-local opt-in surfaces also remain valid where a wrapper category is
  not the right fit. The current maintained example is the large-matrix
  SuiteSparse path in `tests/test_suitesparse.c`, enabled with:

```bash
SPARSE_TEST_LARGE=1 make test
```

That path is live supported test coverage when enabled; it is simply not part
of the default regression run because of fixture/runtime cost.

Historical measurements, retired targets, and old sprint evidence do not stay in
normal suite files as commented-out `RUN_TEST(...)` entries. Preserve that
material in `docs/planning/` artifacts instead of shipping dormant test
scaffolding that overstates current CI coverage.

### Dead-Code Workflow

The dead-code workflow is intentionally separate from `make lint` and
`make test`:

```bash
make deadcode
make deadcode-report
make deadcode-check
```

- `make deadcode` refreshes `build/deadcode-cmake/compile_commands.json`, then
  runs the raw `cppcheck` and `xunused` passes and refreshes the raw evidence
  under `build/deadcode/`
- `make deadcode-report` regenerates those artifacts and writes:
  - `build/deadcode/report.md`
  - `build/deadcode/report.tsv`
- `make deadcode-check` verifies the report-completeness invariants:
  the report exists, every `xunused` finding was categorized, and the
  coverage-gap section is present

Prerequisites:

- `cppcheck` must be installed and on `PATH`
- `xunused` must be installed and on `PATH`

For repository-wide interpretation of the dead-code evidence, completeness
gate, and maintainer cleanup rules, use the
[Maintainer Guide](docs/maintainer_guide.md). Operationally, run the
`deadcode*` targets serially because they share `build/deadcode-cmake` and
`build/deadcode/`. Current platform disposition:

- Linux keeps the dead-code workflow in the enforced quality surface
- macOS keeps dead-code staged pending fresh measurement
- Windows keeps dead-code staged rather than claiming reviewed parity it does
  not yet enforce

### Reviewed Local Quality Path

Reviewed local wrappers sit above the existing direct quality commands:

```bash
make quality-review-compile
make quality-review
make quality-review-full
make quality-review-cmake-compile
make quality-review-cmake
```

- `quality-review-compile` / `quality-review` are the reviewed Makefile path
- `quality-review-full` is the strongest local reviewed baseline command
- `quality-review-cmake-compile` / `quality-review-cmake` are the reviewed
  CMake parity path for clean rebuild + `ctest -N` + full `ctest`
- the CMake wrappers are additive; they do **not** replace the
  Makefile-authoritative formatter, static-analysis, or dead-code checks
- for exact wrapper expansion, rerun guidance, and maintainer-policy
  interpretation, use `make <target>` and the
  [Maintainer Guide](docs/maintainer_guide.md)

### Cross-Platform CI Contract

| Platform | Enforced | Staged | Supplemental / Excluded |
|--------|---------|---------|---------------------------|
| Linux | `make quality-review-compile`; `make quality-review-cmake`; `make deadcode-report`; `make deadcode-check` | none inside the maintained reviewed baseline | direct runtime + `bench-fast`; TSan; coverage |
| macOS | Apple Clang: `make quality-review-compile`; `make quality-review-cmake`; `make wall-check`; `make sanitize` | dead-code (`make deadcode-report`, `make deadcode-check`) pending fresh measurement | Homebrew GCC direct `make` + `make test` + `make wall-check`; supplemental static-first Make install/uninstall + `pkg-config` verification |
| Windows | reviewed CMake configure/build; `ctest -N`; full `ctest` | `make quality-review-compile`; `make quality-review`; dead-code | excluded tests: `test_threads`, `test_sprint4_integration`, `test_fuzz` (so the bounded Sprint 68 property/fuzz lifecycle lane remains outside the reviewed Windows subset); no separate reviewed install-validation lane beyond the CMake-first consumer story |

Use the table above as the compact operator map for enforced, staged, and
supplemental/excluded boundaries. For repository-wide interpretation of those
claims, use the [Maintainer Guide](docs/maintainer_guide.md).

### Quality Readiness Checklist

Use this checklist for a concise release/readiness pass:

- repository-wide warning evidence still uses:
  - `make warning-workflow WARNING_WORKFLOW_LABEL=label`
- strongest local reviewed baseline still passes:
  - `make quality-review-full`
- dead-code evidence refresh and completeness gate still pass:
  - `make deadcode-report`
  - `make deadcode-check`
- reviewed CMake parity still passes when that claim matters:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make quality-review-cmake`
- remaining staged quality/platform limits stay explicit:
  - serialized dead-code execution remains the current operational limit
  - macOS dead-code remains staged pending measurement
  - Windows reviewed-wrapper parity and dead-code remain staged
- docs/examples/header usage stays aligned with shipped behavior
- enforced/staged/excluded platform boundaries still match the
  `Cross-Platform CI Contract` table above

### Maintainer References

For repository-wide quality-contract interpretation, dead-code meaning,
documentation ownership, and stable maintainer norms, use the
[Maintainer Guide](docs/maintainer_guide.md).

For the Sprint 30 authoritative warning-baseline and rebuild references used by
that guide, see:

- [Compile Hygiene Playbook](docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md)
- [Rebuild Workflow](docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md)

Keep README maintainer notes concise and prefer the guide over repeating policy
or `Makefile` target-help detail here.

Tree-mutating local modes are a separate operator category:

- `make sanitize`
- `make asan`
- `make sanitize-all`
- `make tsan`
- `make omp`
- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

These targets intentionally rebuild the shared tree in an alternate mode. When
returning to the normal direct or reviewed path, use:

```bash
make clean
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
├── include/              Public headers
│   ├── sparse_types.h        Error codes, index type (includes sparse_version.h)
│   ├── sparse_version.h      Version macros (generated from VERSION file)
│   ├── sparse_matrix.h       Core data structure, SpMV, block SpMV, I/O
│   ├── sparse_lu.h           LU factorization, solve, block solve
│   ├── sparse_lu_csr.h       CSR LU — scatter-gather elimination, dense blocks
│   ├── sparse_cholesky.h     Cholesky factorization and solve
│   ├── sparse_iterative.h    CG, GMRES, MINRES, BiCGSTAB; block variants; GMRES left/right precond; repeated-run handles for CG/GMRES/MINRES
│   ├── sparse_ilu.h          ILU(0) and ILUT preconditioners
│   ├── sparse_ic.h           IC(0) incomplete Cholesky preconditioner
│   ├── sparse_qr.h           QR factorization, least-squares, rank, null space
│   ├── sparse_dense.h        Dense utilities, Givens, eigensolvers
│   ├── sparse_bidiag.h       Householder bidiagonalization
│   ├── sparse_csr.h          CSR/CSC conversion
│   ├── sparse_reorder.h      Fill-reducing reordering (RCM, AMD, ND, COLAMD)
│   ├── sparse_svd.h          SVD, condition number, pseudoinverse, low-rank
│   └── sparse_vector.h       Dense vector utilities
├── src/                  Library implementation
├── tests/                Unit tests
├── cmake/                CMake config templates
├── examples/             Standalone example programs and CMake integration example
├── benchmarks/           Performance benchmarks and workflow-specific reuse drivers
├── docs/                 Algorithm/format documentation + planning
│   └── planning/         Sprint plans, retrospectives, and project plans
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

After installation, downstream projects can use `pkg-config` or
`find_package(Sparse)` against the same installed static package surface:

- **pkg-config:** `pkg-config --cflags --libs sparse`
- **CMake:** `find_package(Sparse REQUIRED)`, then
  `target_link_libraries(... Sparse::sparse_lu_ortho)`

The maintained package surface is intentionally static-first:

- Unix-like installs produce a static archive such as `libsparse_lu_ortho.a`
- Windows/MSVC installs produce the corresponding static `.lib`
- the exported CMake target and `pkg-config` metadata both describe that same
  static archive surface
- version metadata is single-sourced from `VERSION`
- the exported CMake package version file is exact-version only
- this is a real install/export contract, not a broad shared-library or
  dynamic-ABI guarantee

Focused local proof for that package surface stays explicit:

- `bash tests/test_install.sh` proves the Unix-side Make install/uninstall +
  `pkg-config` path
- `bash tests/test_cmake_install.sh` proves the Unix-side CMake install/export
  + `find_package(Sparse)` path
- macOS CI carries a narrower supplemental Make install/`pkg-config` check
- Windows remains the reviewed CMake-first consumer story rather than a
  separate reviewed install-validation lane

## Documentation

- [Algorithm Description](docs/algorithm.md) — data structure, LU algorithm, complexity analysis
- [Matrix Market Format](docs/matrix_market.md) — supported features and limitations
- [Maintainer Guide](docs/maintainer_guide.md) — repository-wide quality-contract interpretation and documentation ownership
- [Installation Guide](INSTALL.md) — cross-platform build and install instructions

## License

This project is for research and educational purposes.
