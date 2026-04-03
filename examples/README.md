# Examples

Standalone example programs demonstrating the sparse linear algebra library.

## Building

From the project root:

```bash
make examples
```

This builds all examples into the `build/` directory.

## Programs

### example_basic_solve

Solve a 5x5 tridiagonal system `Ax = b` using LU factorization with partial pivoting. Demonstrates matrix creation, factorization, solve, and residual computation.

```bash
./build/example_basic_solve
```

### example_least_squares

Solve an overdetermined 6x3 system via column-pivoted QR factorization. Shows how to find the least-squares solution that minimizes `||Ax - b||` and reports per-equation residuals.

```bash
./build/example_least_squares
```

### example_svd_lowrank

Compute the SVD of an 8x8 matrix and demonstrate low-rank approximation. Shows the singular value spectrum, condition number, rank estimation at different tolerances, and compression ratios for various ranks.

```bash
./build/example_svd_lowrank
```

### example_iterative

Solve a 200x200 sparse system using GMRES with and without ILU(0) preconditioning. Compares iteration counts and convergence behavior.

```bash
./build/example_iterative
```

## Writing Your Own

Each example is a single `.c` file that includes only public headers from `include/`. To compile manually:

```bash
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```
