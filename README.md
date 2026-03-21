# linalg_sparse_orthogonal

A C library for sparse matrices using the **orthogonal linked-list** (cross-linked) representation, with LU decomposition and direct linear system solving.

## Features

- **Orthogonal linked-list storage** — each non-zero is linked into both its row list and column list, enabling efficient row and column traversal
- **Slab pool allocator** with free-list for fast node allocation and reuse
- **LU factorization** with complete or partial pivoting (P·A·Q = L·U)
- **Direct solve** via forward/backward substitution with permutation handling
- **Iterative refinement** to improve solution accuracy
- **Sparse matrix-vector product** (SpMV)
- **Matrix Market I/O** — load and save `.mtx` files (coordinate real general, symmetric, and pattern formats)
- **Drop tolerance** to control fill-in during factorization

## Building

### With Make (recommended)

```bash
make            # build library, tests, and benchmarks
make test       # run all unit tests
make bench      # run benchmarks
make sanitize   # build with address/undefined-behavior sanitizer
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
cc -Iinclude -o example example.c src/*.o -lm
```

## API Overview

The library is split into four headers:

| Header | Purpose |
|--------|---------|
| [`sparse_types.h`](include/sparse_types.h) | `idx_t`, error codes (`sparse_err_t`), pivot strategies |
| [`sparse_matrix.h`](include/sparse_matrix.h) | Sparse matrix lifecycle, element access, SpMV, Matrix Market I/O |
| [`sparse_lu.h`](include/sparse_lu.h) | LU factorization, solve, iterative refinement |
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
- `sparse_lu_solve(mat, b, x)` — solve using factored matrix
- `sparse_lu_refine(A, LU, b, x, max_iters, tol)` — iterative refinement

**I/O:**
- `sparse_save_mm(mat, filename)` / `sparse_load_mm(&mat, filename)` — Matrix Market format

All functions return `sparse_err_t` error codes (except accessors that return values directly). See `sparse_strerror()` for human-readable error messages.

## Performance Characteristics

| Matrix type | Pivoting | Factorization | Fill-in |
|-------------|----------|---------------|---------|
| Tridiagonal (n=5000) | Partial | 0.4 ms | 1.00x (zero fill-in) |
| Tridiagonal (n=5000) | Complete | 256 ms | ~1.6x |
| Arrow (n=200) | Either | — | catastrophic (→ 100% dense) |
| Dense (n=200) | Complete | 1.6 s | 1.00x (already full) |

**Complexity:**
- Partial pivoting: O(nnz) per elimination step — strongly preferred for banded/structured matrices
- Complete pivoting: O(n²) per elimination step due to submatrix search — better numerical stability but much slower
- Solve: O(nnz_LU) for forward/backward substitution
- SpMV: O(nnz)

## Known Limitations

- **Not thread-safe.** All operations are single-threaded.
- **No reordering.** There is no fill-reducing reordering (AMD, RCM, etc.) — fill-in depends entirely on pivoting.
- **Dense vector RHS only.** The solver takes dense vectors for b and x.
- **In-place factorization.** `sparse_lu_factor` overwrites the matrix; always work on a copy if you need the original.
- **No complex or integer matrices.** Only real (double-precision) values are supported.

## Testing

The test suite contains **112 unit tests** with **485 assertions** covering:

- Sparse matrix data structure (31 tests)
- LU factorization and solve (24 tests)
- Matrix Market I/O (18 tests)
- Known reference matrices (15 tests)
- Vector utilities, SpMV, and iterative refinement (24 tests)

```bash
make test    # run all tests
make smoke   # quick smoke test
```

## Project Structure

```
linalg_sparse_orthogonal/
├── include/              Public headers
│   ├── sparse_types.h
│   ├── sparse_matrix.h
│   ├── sparse_lu.h
│   └── sparse_vector.h
├── src/                  Library implementation
│   ├── sparse_types.c
│   ├── sparse_matrix.c
│   ├── sparse_lu.c
│   ├── sparse_vector.c
│   └── sparse_matrix_internal.h
├── tests/                Unit tests
│   ├── test_framework.h
│   ├── test_sparse_matrix.c
│   ├── test_sparse_lu.c
│   ├── test_sparse_io.c
│   ├── test_known_matrices.c
│   ├── test_sparse_vector.c
│   └── data/             Reference .mtx files
├── benchmarks/           Performance benchmarks
├── docs/                 Algorithm and format documentation
├── archive/              Original prototype files
└── planning/             Sprint plans and logs
```

## Documentation

- [Algorithm Description](docs/algorithm.md) — data structure, LU algorithm, complexity analysis
- [Matrix Market Format](docs/matrix_market.md) — supported features and limitations

## License

This project is for research and educational purposes.
