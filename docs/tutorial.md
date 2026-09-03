# Tutorial: linalg_sparse_orthogonal

A practical guide to using the sparse linear algebra library.

## Getting Started

Use this tutorial after the README when you want the fuller learning path. The
first-use route is:

1. build locally;
2. run the first maintained solve;
3. start from CSR, CSC, Matrix Market, or hand-written input;
4. choose the solver family by problem shape;
5. inspect diagnostics from the workflow that produced them;
6. install only when you need a downstream consumer;
7. move to advanced controls, benchmarks, reports, public headers, or
   [API reference](api_reference.md) only after the first workflow works.

For a compact decision tree before you start coding, use the
[solver-selection guide](solver_selection.md). For runnable examples, use
[`examples/README.md`](../examples/README.md). If your data already starts as
CSR, CSC, or Matrix Market input, use the [cookbook](cookbook.md) for the
shortest compressed-first route into direct, iterative, Matrix Market, SVD,
eigensolver, and benchmark handoff workflows.

### Documentation Map

| Need | Use |
|---|---|
| Short project front door | [README.md](../README.md) |
| Runnable first-use examples | [examples/README.md](../examples/README.md) |
| Data-first CSR, CSC, and Matrix Market recipes | [cookbook.md](cookbook.md) |
| Compact problem-shape decision tree | [solver_selection.md](solver_selection.md) |
| Installed package, downstream consumer setup, and support/readiness status | [INSTALL.md](../INSTALL.md) |
| Benchmark commands and generated report indexes | [benchmarks/README.md](../benchmarks/README.md) |
| Exact declarations and ownership contracts | [api_reference.md](api_reference.md) and public headers under [`include/`](../include/) |
| Current algorithm reference | [algorithm.md](algorithm.md) |
| Historical measurement notes | [algorithm_history.md](algorithm_history.md) |
| Maintainer quality policy | [maintainer_guide.md](maintainer_guide.md) |

### Build And Run The First Solve

```bash
make          # build the static library (build/libsparse_lu_ortho.a)
make examples # build example programs
./build/example_basic_solve
```

`example_basic_solve` is the smallest maintained first success path. It shows
matrix creation, a one-shot LU solve, and local residual output. Use
[`examples/README.md#one-shot-direct-example_basic_solve`](../examples/README.md#one-shot-direct-example_basic_solve)
for the runnable-example explanation, or [README.md#quick-start](../README.md#quick-start)
when you want a pasteable standalone program.

Use `make examples-build` when you only need to confirm that maintained
examples compile. Use `make test` when you are validating a code change rather
than learning the first workflow.

### Link Locally Or Install Later

```bash
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```

That command links against the local build-tree static archive. For installed
downstream consumers, use [`INSTALL.md#start-here`](../INSTALL.md#start-here)
instead of copying install/package detail into this tutorial:

- [`INSTALL.md#unix-makepkg-config-consumer`](../INSTALL.md#unix-makepkg-config-consumer)
  for Unix Make/`pkg-config` installed consumers;
- [`INSTALL.md#cmake-consumer`](../INSTALL.md#cmake-consumer) for CMake
  installed consumers.

The maintained package story is static-first and owned by `INSTALL.md`; use
its [support/readiness matrix](../INSTALL.md#support-readiness-matrix) for
current support boundaries. This tutorial only shows the local build-tree path.

Include the headers needed by the workflow you are using:

```c
#include "sparse_matrix.h"   // Core matrix operations
#include "sparse_csr.h"      // CSR/CSC conversion and compressed construction
#include "sparse_lu.h"       // LU factorization
#include "sparse_cholesky.h" // Cholesky factorization (SPD matrices)
#include "sparse_ldlt.h"     // LDLT factorization (symmetric indefinite)
#include "sparse_qr.h"       // QR factorization
#include "sparse_iterative.h" // CG, GMRES, MINRES iterative solvers
#include "sparse_ilu.h"      // ILU preconditioners
#include "sparse_ic.h"       // IC(0) preconditioners
#include "sparse_svd.h"      // SVD, condition number, pseudoinverse
#include "sparse_eigs.h"     // Symmetric sparse eigensolver
```

For exact declarations, option structs, result fields, and ownership
contracts, use [api_reference.md](api_reference.md) and the public headers
under [`include/`](../include/).

---

## 1. Start From Your Matrix

Choose the input route before choosing a solver. Every route below produces a
normal public `SparseMatrix *` that later direct, iterative, QR, SVD, and
eigensolver workflows can consume.

| Starting data | First public step | Runnable or owner reference |
|---|---|---|
| Small hand-written matrix | `sparse_create(...)` plus `sparse_insert(...)` | this tutorial and `example_basic_solve` |
| Caller-owned CSR arrays | `sparse_create_from_csr(...)` or `sparse_from_csr(...)` | [`example_compressed_input`](../examples/README.md#compressed-input-example_compressed_input) and [cookbook](cookbook.md#start-from-your-data) |
| Caller-owned CSC arrays | `sparse_create_from_csc(...)` or `sparse_from_csc(...)` | [`example_compressed_input`](../examples/README.md#compressed-input-example_compressed_input) and [cookbook](cookbook.md#start-from-your-data) |
| Matrix Market file | `sparse_load_mm(...)` | [`example_matrix_market`](../examples/README.md#matrix-market-loaduse-example_matrix_market) and [matrix_market.md](matrix_market.md) |

CSR and CSC constructors validate and copy caller-owned arrays. The caller
still owns the input arrays, and the returned matrix is freed with
`sparse_free(...)`. Use `sparse_create_from_*` when a `NULL` result is enough
for invalid input. Use `sparse_from_*` when the call site needs an explicit
`sparse_err_t` diagnostic.

### Mutable Construction for Small Hand-Written Matrices

```c
#include "sparse_matrix.h"

// Create a 5x5 sparse matrix
SparseMatrix *A = sparse_create(5, 5);

// Insert entries (duplicates overwrite)
sparse_insert(A, 0, 0, 4.0);
sparse_insert(A, 0, 1, -1.0);
sparse_insert(A, 1, 0, -1.0);
sparse_insert(A, 1, 1, 4.0);

// Read entries
double val = sparse_get(A, 0, 0);  // Returns 4.0
double zero = sparse_get(A, 2, 3); // Returns 0.0 (not stored)

// Matrix info
idx_t rows = sparse_rows(A);  // 5
idx_t cols = sparse_cols(A);  // 5
idx_t nnz  = sparse_nnz(A);  // 4
```

### Loading from Matrix Market Files

```c
SparseMatrix *A = NULL;
sparse_err_t err = sparse_load_mm(&A, "matrix.mtx");
if (err != SPARSE_OK) {
    fprintf(stderr, "Load failed: %s\n", sparse_strerror(err));
    if (err == SPARSE_ERR_IO) {
        fprintf(stderr, "system errno: %d\n", sparse_errno());
    }
    return 1;
}
// ... use A ...
sparse_free(A);
```

Loaded Matrix Market matrices are caller-owned `SparseMatrix` objects. Free
them with `sparse_free(...)`. The public Matrix Market surface is
`sparse_load_mm(...)` and `sparse_save_mm(...)`; format details, duplicate
handling, pattern defaults, symmetric expansion, and errno behavior live in
[matrix_market.md](matrix_market.md).

### Matrix Operations

```c
// Copy
SparseMatrix *B = sparse_copy(A);

// Transpose
SparseMatrix *At = sparse_transpose(A);

// Matrix-vector multiply: y = A*x
double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
double y[5] = {0};
sparse_matvec(A, x, y);

// Arithmetic
SparseMatrix *C = NULL;
sparse_add(A, B, 1.0, -1.0, &C);  // C = A - B
sparse_scale(A, 2.0);              // A *= 2

// Always free when done
sparse_free(A);
sparse_free(B);
sparse_free(At);
sparse_free(C);
```

---

### Choose the Solver Workflow

Start with the smallest public workflow that matches the problem, then use the
linked runnable example when you need a concrete reference.

| Need | First workflow | Runnable anchor |
|---|---|---|
| One general square solve | LU | [`example_basic_solve`](../examples/README.md#one-shot-direct-example_basic_solve) |
| Symmetric positive-definite direct solve | Cholesky | this tutorial, then [solver-selection](solver_selection.md#direct-solvers) |
| Symmetric indefinite direct solve | LDLT | [`example_ldlt`](../examples/README.md#symmetric-indefinite-direct-example_ldlt) |
| Rectangular or rank-sensitive solve | QR | [`example_least_squares`](../examples/README.md#rectangular-least-squares-example_least_squares) or [`example_minnorm`](../examples/README.md#underdetermined-minimum-norm-example_minnorm) |
| Large or memory-sensitive linear system | Iterative solver | [`example_iterative`](../examples/README.md#one-shot-iterative-example_iterative) or [`example_ic_minres`](../examples/README.md#ic0-cg-and-minres-example_ic_minres) |
| Symmetric eigenpairs | `sparse_eigs_sym(...)` | [`example_eigs`](../examples/README.md#one-shot-symmetric-eigensolver-example_eigs) |
| Rank, condition, pseudoinverse, or low-rank output | SVD APIs | [`example_svd_lowrank`](../examples/README.md#svd-low-rank-example_svd_lowrank) or [`example_condition`](../examples/README.md#condition-number-example_condition) |
| Procedural operator instead of stored matrix | Matrix-free iterative solver | [`example_matrix_free`](../examples/README.md#matrix-free-iterative-example_matrix_free) |

Use [solver_selection.md](solver_selection.md) for the full decision tree and
evidence boundaries. Examples teach API usage. Benchmarks and report indexes
measure local behavior after you have already chosen the API workflow; they do
not create portable performance or platform claims.

## 2. Direct Solvers

### LU Factorization

Solve `Ax = b` for general square matrices:

```c
#include "sparse_lu.h"

// Factor (modifies the matrix in-place)
SparseMatrix *LU = sparse_copy(A);
sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);

// Solve
double b[] = {1.0, 2.0, 3.0};
double x[3];
sparse_lu_solve(LU, b, x);

// Iterative refinement for higher accuracy
sparse_lu_refine(A, LU, b, x, 3, 1e-15);

sparse_free(LU);
```

Treat LU as a one-shot direct entry point on a fresh matrix or a fresh
`sparse_copy()` of the original coefficients. If you need analyze-once /
factor-many reuse, move to the explicit repeated-run direct lifecycle in
`example_analysis.c` instead of repeatedly re-entering the one-shot LU path.

### Cholesky Factorization

For symmetric positive-definite (SPD) matrices — faster and uses half the storage:

```c
#include "sparse_cholesky.h"

SparseMatrix *L = sparse_copy(A);  // A must be SPD
sparse_err_t err = sparse_cholesky_factor(L);
// err == SPARSE_ERR_NOT_SPD if A is not positive definite

double b[] = {1.0, 2.0, 3.0};
double x[3];
sparse_cholesky_solve(L, b, x);

sparse_free(L);
```

For stable-pattern repeated direct solves, keep the one-shot Cholesky path for
small usage examples and move to the explicit repeated-run direct lifecycle
only when you need analyze-once / factor-many reuse. Start with
`examples/example_analysis.c`. Move to `bench_refactor`,
`bench_refactor_csc`, or `make bench-canonical-report` only when you need
benchmark-side measurement rather than a learning path.

On that explicit repeated-run direct path, failed same-pattern refactors keep
the previous usable factor state intact. Obvious nnz drift is still rejected
as a lifecycle contract violation rather than being treated as an implicit
rebuild request.

For one-shot Cholesky, keep using a fresh matrix or a fresh `sparse_copy()` of
the original coefficients when you still need the original matrix view later.
That keeps the mutation/cancellation caveats local to the working factor copy
instead of the caller's last original view.

### LDL^T Factorization

Use LDL^T for symmetric indefinite systems where Bunch-Kaufman pivoting and
inertia reporting are the right model. For stable-pattern repeated direct
solves, use the same explicit analysis/factor/refactor lifecycle described for
Cholesky rather than treating one-shot factorization as a hidden reuse path.

### QR Factorization

For rectangular or rank-deficient systems. Use the original matrix view here:
QR expects an unfactored, unreordered matrix with identity permutations.
If the matrix may already have been factored or reordered elsewhere, start
from a fresh `sparse_copy()` of the original coefficients before calling QR.
For the broader solver-selection context behind this rule, see the
[solver-selection guide](solver_selection.md).

```c
#include "sparse_qr.h"

sparse_qr_t qr;
sparse_err_t err = sparse_qr_factor(A, &qr);  // A can be m×n with m != n
if (err != SPARSE_OK) {
    // Handle NULL, non-identity permutation, or allocation errors.
    return err;
}

// Numerical rank
idx_t rank = sparse_qr_rank(&qr, 0.0);  // 0.0 = default tolerance

// Least-squares solve. x and residual_norm are caller-owned outputs.
double x[3];
double residual_norm;
err = sparse_qr_solve(&qr, b, x, &residual_norm);
if (err != SPARSE_OK) {
    sparse_qr_free(&qr);
    return err;
}

// Releases factor data stored inside the caller-owned qr object.
sparse_qr_free(&qr);
```

Use `sparse_qr_solve()` for square systems and overdetermined least-squares
problems. For underdetermined systems where you want the minimum 2-norm
solution, call `sparse_qr_solve_minnorm()` instead. Options passed to
minimum-norm solve/refine apply to the temporary QR factorizations those
routines build internally.

---

## 3. Iterative Solvers

### Conjugate Gradient (CG)

For large SPD systems where direct methods are too expensive:

```c
#include "sparse_iterative.h"

sparse_iter_opts_t opts = {
    .max_iter = 1000,
    .tol = 1e-10,
};
sparse_iter_result_t result;
double x[N];
memset(x, 0, N * sizeof(double));  // Initial guess

sparse_err_t err = sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);
printf("Converged in %d iterations, residual = %e\n",
       result.iterations, result.residual_norm);
```

### GMRES

For general (unsymmetric) systems:

```c
sparse_gmres_opts_t opts = {
    .restart = 30,
    .max_iter = 500,
    .tol = 1e-10,
};
sparse_iter_result_t result;
double x[N];
memset(x, 0, N * sizeof(double));

sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
```

For stable-dimension iterative-handle workflows, prepare and reuse an explicit
iterative handle instead of rebuilding scratch state every call. That
repeated-run handle surface is intentionally limited to `CG`, `GMRES`, and
`MINRES`.

### Preconditioning

Choose the preconditioner family to match the matrix class:

- use IC(0) with SPD operators and CG/MINRES workflows
- use ILU(0) or ILUT with GMRES and other general or indefinite-system
  workflows

As in the QR section above, ILU(0), ILUT, and IC(0) expect an original matrix
view with identity permutations. If the matrix may already have been factored
or reordered, build the preconditioner from a fresh `sparse_copy()` of the
original matrix.

Preconditioning can reduce iteration counts on workloads where the
preconditioner matches the matrix assumptions. Treat that as local diagnostic
evidence, not as a portable performance guarantee:

```c
#include "sparse_ilu.h"

// ILU(0) preconditioner
SparseMatrix *A_copy = sparse_copy(A);
sparse_ilu_t ilu;
sparse_ilu_factor(A_copy, &ilu);

// GMRES with left preconditioning
sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);

sparse_ilu_free(&ilu);
sparse_free(A_copy);
```

For more difficult matrices, ILUT with threshold dropping:

```c
SparseMatrix *A_copy = sparse_copy(A);
sparse_ilu_t ilu;
sparse_ilut_opts_t ilu_opts = {
    .tol = 1e-3,
    .max_fill = 10,
};
sparse_ilut_factor(A_copy, &ilu_opts, &ilu);

sparse_ilu_free(&ilu);
sparse_free(A_copy);
```

---

## 4. SVD and Applications

### Full SVD

Compute `A = U * diag(sigma) * V^T`:

As in the QR section above, pass the original unfactored / unreordered matrix
to the SVD routines. If matrix state is uncertain, start from a fresh
`sparse_copy()` of the original coefficients before factoring or reordering
elsewhere.

```c
#include "sparse_svd.h"

// Singular values only
sparse_svd_t svd;
sparse_svd_compute(A, NULL, &svd);
// svd.sigma[0..k-1] in descending order, k = min(m,n)

// With singular vectors (economy/thin SVD)
sparse_svd_opts_t opts = {
    .compute_uv = 1,
    .economy = 1,
};
sparse_svd_compute(A, &opts, &svd);
// svd.U is m×k column-major, svd.Vt is k×n column-major
// Set .economy = 0 to request full U (m×m) and V^T (n×n)

sparse_svd_free(&svd);
```

### Partial SVD (Lanczos)

Compute only the k largest singular values — much faster for large matrices:

```c
idx_t k = 5;
sparse_svd_t svd;
sparse_svd_partial(A, k, NULL, &svd);
// svd.sigma[0..4] are the 5 largest singular values

// With approximate thin singular vectors
sparse_svd_opts_t opts = {
    .compute_uv = 1,
    .economy = 1,
};
sparse_svd_partial(A, k, &opts, &svd);
// Partial SVD supports singular vectors only in the economy/thin path

sparse_svd_free(&svd);
```

Maintained partial-SVD corpus evidence is fixture-local. It currently covers
the clustered/repeated 8x6 lane plus Sprint 151 rank-deficient projector,
sparse low-rank output, and fail-closed recovery rows. Use
[solver_selection.md#svd-and-low-rank-workflows](solver_selection.md#svd-and-low-rank-workflows)
for the current evidence boundary. Do not treat those rows as broad
partial-SVD correctness, raw singular-vector identity, external-library
parity, performance, package/platform/ABI support, or state-of-the-art
behavior.

### Condition Number

```c
sparse_err_t err;
double cond = sparse_cond(A, &err);
// cond = sigma_max / sigma_min
// Returns INFINITY for singular matrices
```

### Numerical Rank

```c
idx_t rank;
sparse_svd_rank(A, 0.0, &rank);  // 0.0 = default tolerance
```

### Pseudoinverse

```c
double *pinv = NULL;
sparse_pinv(A, 0.0, &pinv);
// pinv is n×m column-major dense array
// Satisfies: A * pinv * A ≈ A (Moore-Penrose conditions)
free(pinv);
```

### Low-Rank Approximation

```c
// Dense low-rank
double *lowrank = NULL;
sparse_svd_lowrank(A, rank_k, &lowrank);
// lowrank is m×n column-major dense array
free(lowrank);

// Sparse low-rank (drops small entries)
SparseMatrix *sp_lr = NULL;
sparse_svd_lowrank_sparse(A, rank_k, 0.01, &sp_lr);
sparse_free(sp_lr);
```

---

## 5. Symmetric Eigensolver Workflows

Use `sparse_eigs_sym(...)` when the matrix is symmetric and you need extreme
or near-sigma eigenpairs. Start with the default AUTO backend behavior; choose
an explicit backend only after the basic workflow is understood and local
diagnostics justify it.

```c
#include <stdlib.h>

#include "sparse_eigs.h"

idx_t n = sparse_rows(A);
idx_t k = 5;
sparse_scalar_t *eigenvalues =
    malloc((size_t)k * sizeof(*eigenvalues));
sparse_scalar_t *eigenvectors =
    malloc((size_t)n * (size_t)k * sizeof(*eigenvectors));

sparse_eigs_opts_t opts = {
    .which = SPARSE_EIGS_LARGEST,
    .tol = 1e-10,
    .compute_vectors = 1,
};
sparse_eigs_t result = {
    .eigenvalues = eigenvalues,
    .eigenvectors = eigenvectors,
};
sparse_err_t err = sparse_eigs_sym(A, k, &opts, &result);

free(eigenvectors);
free(eigenvalues);
```

Use [`examples/example_eigs.c`](../examples/example_eigs.c) for the runnable
symmetric eigensolver workflow, including shift-invert and a preconditioned
LOBPCG case. Inspect convergence count, Ritz residual, selected backend, and
peak basis size before changing backend, preconditioner, or shift-invert
settings. Use [`sparse_eigs.h`](../include/sparse_eigs.h) for exact option,
result, backend, and handle details.

This tutorial does not describe a nonsymmetric eigensolver workflow and does
not claim portable state-of-the-art eigensolver parity.

---

## 6. Matrix-Free Interface

Use matrix-free iterative solvers when the operator is too large to store or
is naturally defined procedurally. This is an advanced iterative path after
the standard matrix-shell workflows are understood. For the runnable handoff,
use [`examples/example_matrix_free.c`](../examples/example_matrix_free.c).

```c
#include "sparse_iterative.h"

// Define your operator as a callback
sparse_err_t my_matvec(const void *ctx, idx_t n, const double *x, double *y) {
    // Compute y = A*x without forming A explicitly
    for (idx_t i = 0; i < n; i++) {
        y[i] = 2.0 * x[i];
        if (i > 0)     y[i] -= x[i-1];
        if (i + 1 < n) y[i] -= x[i+1];
    }
    return SPARSE_OK;
}

// Use with CG (SPD operators)
sparse_iter_opts_t cg_opts = {
    .max_iter = 1000,
    .tol = 1e-10,
};
sparse_solve_cg_mf(my_matvec, NULL, n, b, x, &cg_opts, NULL, NULL, &result);

// Use with GMRES (general operators)
sparse_gmres_opts_t gm_opts = {
    .restart = 30,
    .max_iter = 500,
    .tol = 1e-10,
};
sparse_solve_gmres_mf(my_matvec, NULL, n, b, x, &gm_opts, NULL, NULL, &result);
```

---

## Diagnostics Handoff

Diagnostics belong to the workflow that produced them. Start with the smallest
local signal before changing solver family, backend, preconditioner,
tolerance, or benchmark settings.

| Workflow | First diagnostic to inspect |
|---|---|
| CSR/CSC construction | `NULL` result from `sparse_create_from_*` or explicit `sparse_err_t` from `sparse_from_*` |
| Matrix Market input | `sparse_errno()` after `SPARSE_ERR_IO` from `sparse_load_mm(...)` |
| One-shot direct solve | factor/solve return code and problem-local residual |
| Repeated direct lifecycle | analyze/factor/refactor return code, same-pattern invariant, and solve residual |
| Iterative solve | convergence status, residual norm/history, iteration count, stagnation, and breakdown fields |
| QR | rank, residual, nullity/nullspace, and minimum-norm output from QR APIs or examples |
| SVD or partial SVD | rank, condition, triplet residuals, convergence status, and fail-closed status |
| Symmetric eigensolver | Ritz residual, convergence count, selected backend, peak basis size, and shift-invert/preconditioner status |
| Benchmarks or reports | matrix, compiler, backend, thread settings, generated index, and manifest context |

Use [solver_selection.md#diagnostics-handoff](solver_selection.md#diagnostics-handoff)
for the full escalation guide and
[`examples/README.md#diagnostics-handoff`](../examples/README.md#diagnostics-handoff)
for the runnable-example view. Use
[`benchmarks/README.md#reading-benchmark-results`](../benchmarks/README.md#reading-benchmark-results)
before interpreting benchmark or generated report artifacts.

## Advanced Handoffs

After the first workflow works:

- use [README.md#runtime-and-backend-controls](../README.md#runtime-and-backend-controls)
  and [solver_selection.md#advanced-control-escalation](solver_selection.md#advanced-control-escalation)
  for runtime/backend controls;
- use [`benchmarks/README.md`](../benchmarks/README.md) for local benchmark
  commands, generated report indexes, and measurement caveats;
- use [`INSTALL.md`](../INSTALL.md) for installed downstream consumers and
  static-first package support;
- use [api_reference.md](api_reference.md) and public headers under
  [`include/`](../include/) when you need exact declarations, options, result
  structs, ownership rules, or return-code contracts;
- use [`docs/maintainer_guide.md`](maintainer_guide.md) for maintainer
  evidence, report freshness, package/ABI, and support-tier interpretation.

Report freshness commands and normalized report-index checks are maintainer or
advanced-evidence tools. They should not be read as broad performance,
package, platform, external-parity, or release proof.

## Error Handling

All functions return `sparse_err_t`. Common error codes:

| Code | Meaning |
|------|---------|
| `SPARSE_OK` | Success |
| `SPARSE_ERR_NULL` | NULL pointer argument |
| `SPARSE_ERR_ALLOC` | Memory allocation failed |
| `SPARSE_ERR_SINGULAR` | Matrix is singular |
| `SPARSE_ERR_NOT_SPD` | Matrix is not symmetric positive-definite |
| `SPARSE_ERR_NOT_CONVERGED` | Iterative method did not converge |
| `SPARSE_ERR_BADARG` | Invalid argument |
| `SPARSE_ERR_BOUNDS` | Index out of bounds |

Always check return codes in production code:

```c
sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
if (err == SPARSE_ERR_SINGULAR) {
    fprintf(stderr, "Matrix is singular\n");
    // Handle gracefully
}
```
