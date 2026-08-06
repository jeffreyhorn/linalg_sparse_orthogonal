# Examples

Standalone example programs demonstrating the sparse linear algebra library.

## Start Here

Use this file as the compact next-step map after the README front door.

- **Want the smallest first success?**
  - Run `./build/example_basic_solve`.
- **Already have CSR or CSC arrays?**
  - Run `./build/example_compressed_input` for compressed construction into
    the normal public matrix shell, or use the
    [cookbook](../docs/cookbook.md) for direct, iterative, SVD, eigensolver,
    and benchmark handoff paths.
- **Need to choose a solver workflow first?**
  - Use the [solver-selection guide](../docs/solver_selection.md), then return
    here for the matching runnable example.
- **Need the repeated-run direct lifecycle?**
  - Run `./build/example_analysis`.
- **Need a one-shot iterative workflow?**
  - Run `./build/example_iterative`.
- **Need QR least-squares or minimum-norm behavior?**
  - Run `./build/example_least_squares` or `./build/example_minnorm`.
- **Need symmetric-indefinite direct solve behavior?**
  - Run `./build/example_ldlt`.
- **Need the symmetric eigensolver workflow?**
  - Run `./build/example_eigs`.
- **Need SVD, low-rank, or condition-number behavior?**
  - Run `./build/example_svd_lowrank` or `./build/example_condition`.
- **Need to load and use a Matrix Market file?**
  - Run `./build/example_matrix_market` from the project root, then use the
    [cookbook](../docs/cookbook.md) for the load/use handoff.
- **Need an installed downstream-consumer example instead of a local build-tree example?**
  - Use `examples/cmake_example/` and the install flow in
    [INSTALL.md](../INSTALL.md).
- **Need the fuller repeated-run and API walkthrough?**
  - Use the [tutorial](../docs/tutorial.md).
- **Need benchmark/report interpretation after choosing an API workflow?**
  - Use [benchmarks/README.md](../benchmarks/README.md).

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

The public repeated-run handle surface intentionally does not broaden to
`BiCGSTAB` or block iterative workflows.

For the broader user-facing workflow behind those matrix-state rules, start
with the [solver-selection guide](../docs/solver_selection.md), then use the
[tutorial](../docs/tutorial.md) and the relevant public headers under
[`include/`](../include/) when you need deeper API detail. This file stays
focused on example-local behavior and entry points. Use
[benchmarks/README](../benchmarks/README.md) for measurement workflows and the
[Maintainer Guide](../docs/maintainer_guide.md) for quality-policy
interpretation.

For examples that need dynamic scratch buffers, the current small-example
convention is to route allocation through `examples/example_alloc_helpers.h`
rather than open-coding unchecked count/byte multiplication at each call site.

## Programs

### One-Shot Direct: example_basic_solve

Solve a 5x5 tridiagonal system `Ax = b` using LU factorization with partial
pivoting. Demonstrates matrix creation, copying before in-place
factorization, solve, and residual computation.

This is the smallest shipped reference for the one-shot direct rule: if you
still need the original matrix view after factorization, start from a fresh
matrix copy and keep the one-shot mutation on that working factor matrix.

```bash
./build/example_basic_solve
```

Next step after this example:

- stay on the one-shot direct path for small or occasional solves
- move to `example_analysis` when the sparsity pattern is stable across many
  value changes

### Compressed Input: example_compressed_input

Build the same kind of public matrix shell from caller-owned CSR or CSC arrays,
then use the normal one-shot LU workflow. This is the smallest shipped
reference for callers whose data already lives in compressed sparse storage.

The example demonstrates that CSR and CSC arrays are validated and copied, not
adopted. After construction, each returned matrix is freed with
`sparse_free(...)`, while the caller still owns the original compressed arrays.

```bash
./build/example_compressed_input
```

Next step after this example:

- use `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)` when
  `NULL` on invalid input is enough
- use `sparse_from_csr(...)` or `sparse_from_csc(...)` when the call site needs
  explicit `sparse_err_t` diagnostics
- move to `example_analysis` when repeated same-pattern direct solves are the
  reason to manage analysis and factor objects explicitly

### Symmetric Indefinite Direct: example_ldlt

Solve a small KKT-style symmetric indefinite system with LDL^T. Demonstrates
Bunch-Kaufman factorization, inertia reporting, AMD fill-reducing ordering,
iterative refinement, and condition number estimation.

Use this when the problem is square and symmetric but not positive-definite.
For SPD systems, start with Cholesky instead. For rectangular or rank-sensitive
systems, use the QR examples below.

```bash
./build/example_ldlt
```

### Repeated-Run Direct: example_analysis

Demonstrate the explicit repeated-run direct lifecycle:

- zero-init `sparse_analysis_t` / `sparse_factors_t`
- analyze once through `sparse_analyze(...)`
- factor / solve through `sparse_factor_numeric(...)` and
  `sparse_factor_solve(...)`
- refactor / solve many through `sparse_refactor_numeric(...)`

This is the strongest shipped adoption example for stable-pattern repeated
direct solves. It complements the smaller one-shot examples rather than
replacing them.

Use this path when reuse is the point. Keep the smaller LU/Cholesky/LDL^T
examples for occasional one-shot solves and move here only when you want the
explicit analyze / factor / solve / refactor lifecycle.

For measurement after you adopt that lifecycle, move to the benchmark surfaces
rather than expecting examples to double as timing harnesses:

- `bench_refactor` / `bench_refactor_csc` for repeated-run direct reuse
- `bench_iterative_reuse` for iterative handles
- `bench_eigs_reuse` for eigensolver handles
- `make bench-canonical-report` when you want one threshold-free snapshot of
  the canonical maintained benchmark surface

```bash
./build/example_analysis
```

Next step after this example:

- move to the [tutorial](../docs/tutorial.md) for the fuller repeated-run
  workflow and API walkthrough
- move to the benchmark surfaces when you need measurement rather than another
  teaching example

### Rectangular Least-Squares: example_least_squares

Solve an overdetermined 6x3 system via column-pivoted QR factorization. Shows
how to find the least-squares solution that minimizes `||Ax - b||` and reports
per-equation residuals. For underdetermined minimum-2-norm solves, use the
public API path `sparse_qr_solve_minnorm()`, documented in the
[README](../README.md) and [`sparse_qr.h`](../include/sparse_qr.h).
The maintained QR corpus proof for
`qr_rank_deficient_6x4_nullspace_v1` is separate from these teaching examples:
it lives in [`tests/test_qr_corpus.c`](../tests/test_qr_corpus.c) and proves
only fixture-local rank, nullity, and nullspace residual behavior.

```bash
./build/example_least_squares
```

### Underdetermined Minimum-Norm: example_minnorm

Solve an underdetermined system with the public minimum-2-norm QR path. This is
the companion to `example_least_squares` for cases where there are fewer
equations than unknowns and the minimum-norm solution is the desired answer.

```bash
./build/example_minnorm
```

### Reorder / Fill: example_colamd

Compute a COLAMD column ordering for an unsymmetric matrix, compare natural
versus COLAMD LU fill, and use COLAMD with QR factorization. This example is
the local teaching route for column ordering; symmetric fill-reducing
workflows use RCM, AMD, or ND through the reorder and analysis APIs described
in the solver-selection guide.

```bash
./build/example_colamd
```

### SVD / Low-Rank: example_svd_lowrank

Compute the SVD of an 8x8 matrix and demonstrate low-rank approximation. Shows
the singular value spectrum, condition number, rank estimation at different
tolerances, and compression ratios for various ranks.

For partial-SVD edge-case confidence, the maintained corpus has a separate
fixture-local proof for one generated 8x6 clustered/repeated diagonal case with
`k = 3`. That proof lives in `tests/test_svd_partial_corpus.c` and is not part
of the example's user-facing output or a broad performance/parity claim.

```bash
./build/example_svd_lowrank
```

### Condition Number: example_condition

Compare well-conditioned, ill-conditioned, and singular systems with
`sparse_cond(...)` and a small one-shot LU solve. Use this example when you
need the conditioning workflow rather than the full low-rank SVD workflow.

```bash
./build/example_condition
```

### One-Shot Iterative: example_iterative

Solve a 200x200 sparse system using GMRES with and without ILU(0)
preconditioning. Compares iteration counts and convergence behavior, and builds
the ILU(0) preconditioner from a fresh matrix copy so the original matrix view
remains available to the iterative solve.

This stays on the one-shot path by design. For stable-dimension repeated runs,
the public iterative-handle path now covers `CG`, `GMRES`, and `MINRES`.

```bash
./build/example_iterative
```

### IC(0), CG, and MINRES: example_ic_minres

Demonstrate IC(0)-preconditioned CG on an SPD system and MINRES on symmetric
indefinite systems. This is the strongest local example for matching
preconditioner assumptions to iterative solver assumptions.

```bash
./build/example_ic_minres
```

### Matrix-Free Iterative: example_matrix_free

Solve with GMRES using a custom matrix-vector callback instead of forming a
stored sparse matrix. Use this route for structured operators where a public
matrix shell is not the natural starting point.

```bash
./build/example_matrix_free
```

### Matrix Market Load/Use: example_matrix_market

Load `tests/data/tridiagonal_20.mtx` with `sparse_load_mm(...)`, build a
right-hand side from the loaded matrix, solve through the normal one-shot LU
workflow, and free the loaded matrix with `sparse_free(...)`. Run this example
from the project root so the test-data path resolves.

This example treats Matrix Market as public load/use functions, not as a
separate public Matrix I/O module or builder API. I/O failures report
`SPARSE_ERR_IO` and expose the captured system errno through `sparse_errno()`.

```bash
./build/example_matrix_market
```

### One-Shot Symmetric Eigensolver: example_eigs

Compute symmetric eigenpairs with `sparse_eigs_sym` across three representative
workflows. Part (a) finds the five largest eigenvalues of a small SPD
SuiteSparse matrix (nos4, n = 100) and reports per-pair eigen-equation
residuals. Part (b) exercises shift-invert mode: three eigenvalues nearest
σ = 0 on a KKT indefinite saddle-point matrix, composing with the LDL^T path.
Part (c) runs explicit LOBPCG with IC(0) preconditioning on `bcsstk04`. Run
from the project root so the SuiteSparse fixtures resolve.

This example still uses the one-shot eigensolver entry point by design. For
stable-dimension repeated runs, the public eigensolver handle path covers
grow-m Lanczos, thick-restart Lanczos, and explicit `LOBPCG`.

```bash
./build/example_eigs
```

### Installed Consumer Example: examples/cmake_example

Use `examples/cmake_example/` when you want the downstream installed-consumer
story rather than a local build-tree example. This example stays separate from
the local `./build/example_*` teaching binaries because it demonstrates the
installed CMake consumer path instead of the local adoption flow.

For that installed-consumer path, use:

- `examples/cmake_example/CMakeLists.txt`
- `examples/cmake_example/main.c`
- [INSTALL.md](../INSTALL.md)

## Writing Your Own

Each build-tree teaching example is a single `.c` file that includes only public
headers from `include/`. To compile manually:

```bash
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```
