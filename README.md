# linalg_sparse_orthogonal

A C library for sparse matrices using the **orthogonal linked-list** (cross-linked) representation, with direct and iterative linear system solvers.

## Features

### Core Data Structure
- **Orthogonal linked-list storage** — each non-zero is linked into both its row list and column list, enabling efficient row and column traversal
- **Slab pool allocator** with free-list for fast node allocation and reuse

### Direct Solvers
- **LU factorization** with complete or partial pivoting (P·A·Q = L·U)
- **Cholesky factorization** for symmetric positive-definite matrices (A = L·L^T, ~50% less storage than LU)
- **Direct solve** via forward/backward substitution with permutation handling
- **Iterative refinement** to improve solution accuracy

### Iterative Solvers
- **Conjugate Gradient (CG)** for SPD systems with optional preconditioning
- **Restarted GMRES(k)** for general unsymmetric systems with left preconditioning
- **ILU(0) preconditioner** — incomplete LU with no fill-in, 3-1000× iteration reduction

### Matrix Operations
- **Sparse matrix-vector product** (SpMV) with optional OpenMP parallelization
- **Sparse matrix-matrix multiply** — C = A*B via Gustavson's algorithm (`sparse_matmul`)
- **Matrix arithmetic** — scalar scaling (`sparse_scale`) and addition (`sparse_add`)
- **Infinity norm** with internal caching (`sparse_norminf`)

### Reordering & Preconditioning
- **Fill-reducing reordering** — Reverse Cuthill-McKee (RCM) and Approximate Minimum Degree (AMD)
- **Condition number estimation** — Hager/Higham 1-norm estimator from LU factors

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
make            # build library, tests, and benchmarks
make test       # run all unit tests
make bench      # run benchmarks
make omp        # build and test with OpenMP-enabled parallel SpMV
make sanitize   # build with undefined-behavior sanitizer
make clean      # remove build artifacts
```

### With CMake

```bash
mkdir build && cd build
cmake ..
make
ctest           # run tests
```

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
    sparse_load_mm(&A, "matrix.mtx");
    int n = sparse_rows(A);

    double *b = malloc(n * sizeof(double));
    double *x = calloc(n, sizeof(double));  /* zero initial guess */
    /* ... set up b ... */

    /* ILU(0) preconditioned GMRES */
    sparse_ilu_t ilu;
    sparse_ilu_factor(A, &ilu);

    sparse_gmres_opts_t opts = { .max_iter = 1000, .restart = 50, .tol = 1e-10 };
    sparse_iter_result_t result;
    sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);

    printf("Converged in %d iterations, residual = %e\n",
           result.iterations, result.residual_norm);

    sparse_ilu_free(&ilu);
    free(b); free(x);
    sparse_free(A);
}
```

## API Overview

| Header | Purpose |
|--------|---------|
| [`sparse_types.h`](include/sparse_types.h) | `idx_t`, error codes (`sparse_err_t`), pivot/reorder strategies |
| [`sparse_matrix.h`](include/sparse_matrix.h) | Sparse matrix lifecycle, element access, SpMV, Matrix Market I/O |
| [`sparse_lu.h`](include/sparse_lu.h) | LU factorization, solve, condition estimation, iterative refinement |
| [`sparse_cholesky.h`](include/sparse_cholesky.h) | Cholesky factorization and solve for SPD matrices |
| [`sparse_iterative.h`](include/sparse_iterative.h) | CG and GMRES iterative solvers with preconditioner support |
| [`sparse_ilu.h`](include/sparse_ilu.h) | ILU(0) incomplete factorization preconditioner |
| [`sparse_csr.h`](include/sparse_csr.h) | CSR/CSC compressed format conversion |
| [`sparse_reorder.h`](include/sparse_reorder.h) | Fill-reducing reordering (RCM, AMD), permutation, bandwidth |
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

**Cholesky (SPD matrices):**
- `sparse_cholesky_factor(mat)` — in-place A = L·L^T
- `sparse_cholesky_factor_opts(mat, &opts)` — with optional AMD/RCM reordering
- `sparse_cholesky_solve(mat, b, x)` — solve using Cholesky factors

**Iterative solvers:**
- `sparse_solve_cg(A, b, x, &opts, precond, ctx, &result)` — Preconditioned Conjugate Gradient (SPD only)
- `sparse_solve_gmres(A, b, x, &opts, precond, ctx, &result)` — Restarted GMRES(k) with left preconditioning

**ILU(0) preconditioner:**
- `sparse_ilu_factor(A, &ilu)` — ILU(0) factorization (no fill-in beyond A's pattern)
- `sparse_ilu_solve(&ilu, r, z)` — apply preconditioner: solve L*U*z = r
- `sparse_ilu_precond` — callback compatible with `sparse_precond_fn`
- `sparse_ilu_free(&ilu)` — free ILU factors

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

**Complexity:**
- Partial pivoting: O(nnz) per elimination step — strongly preferred for banded/structured matrices
- Complete pivoting: O(n²) per elimination step due to submatrix search — better numerical stability but much slower
- Solve: O(nnz_LU) for forward/backward substitution
- SpMV: O(nnz)

## Thread Safety

The library is safe for concurrent use under the following contract:

| Operation | Thread-safe? | Notes |
|-----------|:---:|-------|
| Concurrent solves on the same factored matrix | Yes | Solve is pure read-only on the matrix |
| Concurrent factorization of different matrices | Yes | Each matrix has its own pool allocator |
| Concurrent read-only access (nnz, get, matvec) | Yes | No shared mutable state |
| `sparse_errno()` | Yes | Uses `_Thread_local` storage |
| Concurrent mutation of the same matrix | **No** | Insert/remove/factor on a shared matrix requires external synchronization |

**Optional mutex support:** Compile with `-DSPARSE_MUTEX` and `-pthread` to add per-matrix mutex locking on `sparse_insert()` and `sparse_remove()`. This serializes concurrent insert/remove calls on the same matrix. Note: factorization (`sparse_lu_factor`, `sparse_cholesky_factor`) is not mutex-protected and must not be called concurrently on the same matrix. Not recommended — prefer separate matrices per thread.

## Known Limitations

- **Dense vector RHS only.** The solver takes dense vectors for b and x.
- **In-place factorization.** `sparse_lu_factor` overwrites the matrix; always work on a copy if you need the original.
- **No complex or integer matrices.** Only real (double-precision) values are supported.

## Testing

The test suite contains **406 unit tests** across 19 test suites:

- Sparse matrix data structure, norms, and symmetry check (43 tests)
- LU factorization, solve, transpose solve, and condition estimation (37 tests)
- Matrix Market I/O with errno validation (22 tests)
- Known reference matrices (15 tests)
- Vector utilities, SpMV, and iterative refinement (24 tests)
- Edge cases and relative tolerance hardening (24 tests)
- Integration tests (7 tests)
- Matrix arithmetic — scale and add (23 tests)
- SuiteSparse real-world matrix validation (10 tests)
- Reordering — permute, bandwidth, RCM, AMD, factor_opts integration (38 tests)
- Cholesky factorization and solve (21 tests)
- CSR/CSC conversion and round-trip (11 tests)
- Sparse matrix-matrix multiply (14 tests)
- Thread safety — concurrent solve and insert (7 tests)
- Sprint 4 cross-feature integration (5 tests)
- Iterative solvers — CG, GMRES, convergence, SuiteSparse validation (62 tests)
- ILU(0) preconditioner — factorization, solve, integration (18 tests)
- Parallel SpMV — correctness, reproducibility, solver integration (12 tests)
- Sprint 5 cross-feature integration (9 tests)

```bash
make test          # run all tests
make smoke         # quick smoke test
make sanitize      # UBSan (undefined behavior)
make asan          # ASan (address sanitizer) — requires GCC or LLVM clang on macOS
make sanitize-all  # both ASan + UBSan
make tsan          # TSan (thread sanitizer) for concurrent tests
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
├── include/              Public headers (9 headers)
│   ├── sparse_types.h
│   ├── sparse_matrix.h
│   ├── sparse_lu.h
│   ├── sparse_cholesky.h
│   ├── sparse_iterative.h
│   ├── sparse_ilu.h
│   ├── sparse_csr.h
│   ├── sparse_reorder.h
│   └── sparse_vector.h
├── src/                  Library implementation (9 source + 1 internal header)
│   ├── sparse_types.c
│   ├── sparse_matrix.c
│   ├── sparse_lu.c
│   ├── sparse_cholesky.c
│   ├── sparse_iterative.c
│   ├── sparse_ilu.c
│   ├── sparse_csr.c
│   ├── sparse_reorder.c
│   ├── sparse_vector.c
│   └── sparse_matrix_internal.h
├── tests/                Unit tests (19 suites, 401 tests)
│   ├── test_framework.h
│   ├── test_sparse_matrix.c
│   ├── test_sparse_lu.c
│   ├── test_sparse_io.c
│   ├── test_known_matrices.c
│   ├── test_sparse_vector.c
│   ├── test_edge_cases.c
│   ├── test_integration.c
│   ├── test_sparse_arith.c
│   ├── test_suitesparse.c
│   ├── test_reorder.c
│   ├── test_cholesky.c
│   ├── test_csr.c
│   ├── test_matmul.c
│   ├── test_threads.c
│   ├── test_sprint4_integration.c
│   ├── test_iterative.c
│   ├── test_ilu.c
│   ├── test_omp.c
│   ├── test_sprint5_integration.c
│   └── data/             Reference .mtx files (8 + 6 SuiteSparse)
├── benchmarks/           Performance benchmarks (4 programs)
├── docs/                 Algorithm and format documentation
├── archive/              Original prototype files
└── planning/             Sprint plans and logs
```

## Documentation

- [Algorithm Description](docs/algorithm.md) — data structure, LU algorithm, complexity analysis
- [Matrix Market Format](docs/matrix_market.md) — supported features and limitations

## License

This project is for research and educational purposes.
