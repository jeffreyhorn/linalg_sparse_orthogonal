# Examples

Standalone example programs demonstrating the sparse linear algebra library.

## Building

From the project root:

```bash
make examples
```

This builds all examples into the `build/` directory.

The examples are intentionally small public-usage references. When an example
demonstrates an in-place factorization or an incomplete-factorization
preconditioner, it uses a fresh matrix copy before mutating factor state so
the original matrix view remains available where the API expects it.

The shipped examples still lean on the one-shot public APIs because those
remain first-class and are the simplest entry point for most callers. The
explicit repeated-run public surfaces are opt-in paths for stable-dimension
repeated runs, not replacements for the small one-shot examples here:

- iterative handles:
  - `CG`
  - `GMRES`
  - `MINRES`
- symmetric eigensolver handle:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`

Sprint 54 intentionally does not broaden that handle surface to `BiCGSTAB` or
block iterative workflows.

For examples that need dynamic scratch buffers, the current small-example
convention is to route allocation through `examples/example_alloc_helpers.h`
rather than open-coding unchecked count/byte multiplication at each call site.

For the broader user-facing workflow behind those matrix-state rules, use the
[tutorial](../docs/tutorial.md) and the relevant public headers under
[`include/`](../include/). This file stays focused on example-local behavior
and entry points.

## Programs

### example_basic_solve

Solve a 5x5 tridiagonal system `Ax = b` using LU factorization with partial
pivoting. Demonstrates matrix creation, copying before in-place
factorization, solve, and residual computation.

```bash
./build/example_basic_solve
```

### example_analysis

Demonstrate the explicit repeated-run direct lifecycle:

- zero-init `sparse_analysis_t` / `sparse_factors_t`
- analyze once through `sparse_analyze(...)`
- factor / solve through `sparse_factor_numeric(...)` and
  `sparse_factor_solve(...)`
- refactor / solve many through `sparse_refactor_numeric(...)`

This is the strongest shipped example for stable-pattern repeated direct
solves. It complements the smaller one-shot examples rather than replacing
them.

```bash
./build/example_analysis
```

### example_least_squares

Solve an overdetermined 6x3 system via column-pivoted QR factorization. Shows
how to find the least-squares solution that minimizes `||Ax - b||` and reports
per-equation residuals. For underdetermined minimum-2-norm solves, use the
public API path `sparse_qr_solve_minnorm()`, documented in the
[README](../README.md) and [`sparse_qr.h`](../include/sparse_qr.h).

```bash
./build/example_least_squares
```

### example_svd_lowrank

Compute the SVD of an 8x8 matrix and demonstrate low-rank approximation. Shows the singular value spectrum, condition number, rank estimation at different tolerances, and compression ratios for various ranks.

```bash
./build/example_svd_lowrank
```

### example_iterative

Solve a 200x200 sparse system using GMRES with and without ILU(0)
preconditioning. Compares iteration counts and convergence behavior, and builds
the ILU(0) preconditioner from a fresh matrix copy so the original matrix view
remains available to the iterative solve.

This stays on the one-shot path by design. For stable-dimension repeated runs,
the public iterative-handle path now covers `CG`, `GMRES`, and `MINRES`.

```bash
./build/example_iterative
```

### example_eigs

Compute symmetric eigenpairs with `sparse_eigs_sym` (Sprint 20). Part (a)
finds the five largest eigenvalues of a small SPD SuiteSparse matrix (nos4,
n = 100) and reports per-pair eigen-equation residuals. Part (b) exercises
shift-invert mode: three eigenvalues nearest σ = 0 on a KKT indefinite
saddle-point matrix, composing with the LDL^T dispatch from Sprint 20 Days 4-6.
Part (c) runs explicit LOBPCG with IC(0) preconditioning on `bcsstk04`. Run
from the project root so the SuiteSparse fixtures resolve.

This example still uses the one-shot eigensolver entry point by design. For
stable-dimension repeated runs, the public eigensolver handle path covers
grow-m Lanczos, thick-restart Lanczos, and explicit `LOBPCG`.

```bash
./build/example_eigs
```

## Writing Your Own

Each example is a single `.c` file that includes only public headers from `include/`. To compile manually:

```bash
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```
