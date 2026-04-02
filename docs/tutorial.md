# Tutorial: linalg_sparse_orthogonal

A practical guide to using the sparse linear algebra library.

## Getting Started

### Building the Library

```bash
make          # Build the static library (build/libsparse_lu_ortho.a)
make test     # Run all tests
make examples # Build example programs
```

### Linking Your Program

```bash
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```

Include the headers you need:

```c
#include "sparse_matrix.h"   // Core matrix operations
#include "sparse_lu.h"       // LU factorization
#include "sparse_cholesky.h" // Cholesky factorization (SPD matrices)
#include "sparse_qr.h"       // QR factorization
#include "sparse_iterative.h" // CG, GMRES iterative solvers
#include "sparse_ilu.h"      // ILU preconditioners
#include "sparse_svd.h"      // SVD, condition number, pseudoinverse
```

---

## 1. Creating and Manipulating Sparse Matrices

### Creating a Matrix

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
    // Handle error
}
// ... use A ...
sparse_free(A);
```

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
sparse_lu_refine(A, LU, b, x, 3);

sparse_free(LU);
```

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

### QR Factorization

For rectangular or rank-deficient systems:

```c
#include "sparse_qr.h"

sparse_qr_t qr;
sparse_qr_factor(A, &qr);  // A can be m×n with m != n

// Numerical rank
idx_t rank = sparse_qr_rank(&qr, 0.0);  // 0.0 = default tolerance

// Least-squares solve (minimizes ||Ax - b||)
double x[3];
double residual_norm;
sparse_qr_solve(&qr, b, x, &residual_norm);

sparse_qr_free(&qr);
```

---

## 3. Iterative Solvers

### Conjugate Gradient (CG)

For large SPD systems where direct methods are too expensive:

```c
#include "sparse_iterative.h"

sparse_cg_opts_t opts = {.max_iter = 1000, .tol = 1e-10};
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
sparse_gmres_opts_t opts = {.restart = 30, .max_iter = 500, .tol = 1e-10};
sparse_iter_result_t result;
double x[N];
memset(x, 0, N * sizeof(double));

sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
```

### Preconditioning

ILU preconditioning dramatically reduces iteration counts:

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
sparse_ilu_opts_t ilu_opts = {.droptol = 1e-3, .fillfactor = 10};
sparse_ilut_factor(A_copy, &ilu, &ilu_opts);
```

---

## 4. SVD and Applications

### Full SVD

Compute `A = U * diag(sigma) * V^T`:

```c
#include "sparse_svd.h"

// Singular values only
sparse_svd_t svd;
sparse_svd_compute(A, NULL, &svd);
// svd.sigma[0..k-1] in descending order, k = min(m,n)

// With singular vectors (economy/thin SVD)
sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
sparse_svd_compute(A, &opts, &svd);
// svd.U is m×k column-major, svd.Vt is k×n column-major

sparse_svd_free(&svd);
```

### Partial SVD (Lanczos)

Compute only the k largest singular values — much faster for large matrices:

```c
idx_t k = 5;
sparse_svd_t svd;
sparse_svd_partial(A, k, NULL, &svd);
// svd.sigma[0..4] are the 5 largest singular values

// With approximate singular vectors
sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
sparse_svd_partial(A, k, &opts, &svd);

sparse_svd_free(&svd);
```

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

## 5. Matrix-Free Interface

For operators too large to store or defined procedurally:

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
sparse_cg_opts_t cg_opts = {.max_iter = 1000, .tol = 1e-10};
sparse_solve_cg_mf(my_matvec, NULL, n, b, x, &cg_opts, NULL, NULL, &result);

// Use with GMRES (general operators)
sparse_gmres_opts_t gm_opts = {.restart = 30, .max_iter = 500, .tol = 1e-10};
sparse_solve_gmres_mf(my_matvec, NULL, n, b, x, &gm_opts, NULL, NULL, &result);
```

---

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
