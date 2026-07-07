# Solver Selection Guide

Use this guide when you know the shape of your sparse problem and need to
choose the smallest supported public workflow. Start from the way your matrix
arrives, then choose the solver family whose assumptions match the problem.

For runnable examples, use [`examples/README.md`](../examples/README.md). For
API details, use the public headers under [`include/`](../include/). For local
performance measurement, use [`benchmarks/README.md`](../benchmarks/README.md).

## Start From Your Matrix

| Starting point | Use this public path | Ownership and cleanup |
|---|---|---|
| Caller-owned CSR arrays | `sparse_create_from_csr(...)` for simple construction, or `sparse_from_csr(...)` for explicit `sparse_err_t` diagnostics. | Input arrays remain caller-owned. The returned `SparseMatrix *` is independent and is freed with `sparse_free(...)`. |
| Caller-owned CSC arrays | `sparse_create_from_csc(...)` or `sparse_from_csc(...)`. | Input arrays remain caller-owned. The returned `SparseMatrix *` is independent and is freed with `sparse_free(...)`. |
| Matrix Market file | `sparse_load_mm(...)`. | The loaded matrix is caller-owned and freed with `sparse_free(...)`. I/O failures expose system errno through `sparse_errno()`. |
| Small hand-written matrix | `sparse_create(...)` plus `sparse_insert(...)`. | This is best for small examples and tests, not for bulk imported data. |
| Existing matrix shell | Public copy, export, transpose, matrix operation, and solver APIs. | Factorization may mutate working matrices, so use `sparse_copy(...)` when you still need the original coefficients. |

If your data already lives in CSR or CSC, prefer the compressed-first
constructors instead of inserting every entry manually. The compressed arrays
are validated and copied into the normal public matrix shell; the library does
not adopt those arrays.

## Choose the Smallest Workflow

| Need | First workflow to try |
|---|---|
| Solve one general square system | LU |
| Solve one symmetric positive-definite system | Cholesky |
| Solve one symmetric indefinite system | LDLT |
| Solve least-squares, rectangular, or rank-sensitive systems | QR |
| Solve many systems with the same sparsity pattern and changing values | Explicit analysis/factor/refactor lifecycle |
| Solve large systems where direct solve cost or memory is the issue | Iterative solver with diagnostics and optional preconditioning |
| Compute symmetric eigenpairs | `sparse_eigs_sym(...)` |
| Compute rank, condition, pseudoinverse, or low-rank approximations | SVD APIs |
| Compare local runtime or fill behavior | Benchmarks, after the API workflow is chosen |

Examples teach API usage. Benchmarks measure local workflow behavior. Treat
benchmark output as branch-local and configuration-sensitive, not as a
portable timing guarantee.

## Direct Solvers

| Problem | Use | Notes |
|---|---|---|
| General square matrix | LU | Use a fresh matrix or `sparse_copy(...)` if you need the original later. |
| Symmetric positive-definite matrix | Cholesky | Non-SPD inputs report an error; do not use Cholesky as a general fallback. |
| Symmetric indefinite matrix | LDLT | Use when symmetry is part of the problem model, such as KKT-style systems. |
| Rectangular or rank-sensitive least-squares | QR | Use QR-specific APIs for rectangular, least-squares, minimum-norm, and rank-sensitive workflows. |

Use the explicit repeated-run direct lifecycle when reuse is the point:

1. `sparse_analyze(...)`
2. `sparse_factor_numeric(...)`
3. `sparse_factor_solve(...)`
4. `sparse_refactor_numeric(...)` for later same-pattern value changes

That lifecycle is for stable sparsity patterns. It is not a hidden structural
rebuild path.

Useful starting examples:

- `example_basic_solve` for the smallest LU one-shot solve.
- `example_analysis` for analyze-once / factor-many direct reuse.
- `example_colamd` for QR/COLAMD ordering usage.

Relevant headers:

- [`sparse_lu.h`](../include/sparse_lu.h)
- [`sparse_cholesky.h`](../include/sparse_cholesky.h)
- [`sparse_ldlt.h`](../include/sparse_ldlt.h)
- [`sparse_qr.h`](../include/sparse_qr.h)
- [`sparse_analysis.h`](../include/sparse_analysis.h)

## Reordering and Fill

Use reordering to reduce work or fill, not to change the mathematical problem.

| Need | Public route |
|---|---|
| Symmetric fill reduction | RCM, AMD, or ND through reorder APIs or analysis options. |
| Unsymmetric or QR column ordering | COLAMD. |
| Repeated direct lifecycle reordering | `sparse_analysis_opts_t` reorder settings. |
| Local measurement of ordering choices | `bench_reorder`, `bench_colamd`, and benchmark docs. |

Keep symmetric permutations and column-only permutations separate. RCM, AMD,
and ND are symmetric-ordering tools. COLAMD is the column-ordering route for
unsymmetric/QR workflows.

Relevant header: [`sparse_reorder.h`](../include/sparse_reorder.h).

## Iterative Solvers

| Problem | First choice | Reuse support |
|---|---|---|
| Symmetric positive-definite | CG | Repeated-run handle supported. |
| General unsymmetric | GMRES | Repeated-run handle supported. |
| Symmetric indefinite | MINRES | Repeated-run handle supported. |
| General nonsymmetric compatibility path | BiCGSTAB | One-shot only. |

Use the solver result fields to inspect convergence, residual norm,
stagnation, and breakdown. Use the input `x` vector as the initial guess; pass
a zeroed vector when you want no prior guess.

Preconditioners are acceleration tools, not universal guarantees:

- ILU(0) and ILUT are the general/nonsymmetric preconditioner family.
- IC(0) is the symmetric positive-definite preconditioner family.
- Match the preconditioner to the solver assumptions.

Useful starting example:

- `example_iterative` for one-shot GMRES with and without ILU(0).

Relevant headers:

- [`sparse_iterative.h`](../include/sparse_iterative.h)
- [`sparse_ilu.h`](../include/sparse_ilu.h)
- [`sparse_ic.h`](../include/sparse_ic.h)

## Eigensolver Workflows

Use `sparse_eigs_sym(...)` for symmetric sparse eigensolver workflows. The
default `SPARSE_EIGS_BACKEND_AUTO` is the normal starting point; explicit
backend selection is for profiling or workload-specific control.

Use repeated-run eigensolver handles when the dimension is stable and
workspace reuse matters. Shift-invert and preconditioning are advanced paths
with solver-specific requirements.

Useful starting example:

- `example_eigs` for symmetric eigenpairs, shift-invert, and an explicit
  LOBPCG/preconditioner case.

Relevant header: [`sparse_eigs.h`](../include/sparse_eigs.h).

This guide does not claim nonsymmetric eigensolver support or portable
state-of-the-art parity. Use benchmark output as local measurement context.

## SVD and Low-Rank Workflows

Use SVD APIs when you need singular values, numerical rank, condition
estimates, pseudoinverse behavior, or low-rank approximations. Treat dense
outputs and low-rank buffers according to the ownership rules of the public
SVD APIs; do not depend on private dense workspaces.

Useful starting example:

- `example_svd_lowrank` for singular values, rank, condition estimate, and
  low-rank approximation.

Relevant headers:

- [`sparse_svd.h`](../include/sparse_svd.h)
- [`sparse_bidiag.h`](../include/sparse_bidiag.h)

## Matrix Market Inputs

Use `sparse_load_mm(...)` when your matrix arrives as a Matrix Market file.
Then use the returned `SparseMatrix *` with the same solver-selection rules as
any other public matrix shell.

Current public docs for format support live in
[`docs/matrix_market.md`](matrix_market.md). Use `example_matrix_market` for a
small load/use workflow. The format guide owns duplicate-entry, zero-elision,
ownership, pattern, symmetric-expansion, errno, and runtime wording.

Do not describe this as a public Matrix I/O module or public builder API. The
public surface is the load/save functions declared in
[`sparse_matrix.h`](../include/sparse_matrix.h).

## Example Handoff

| Question | Start here |
|---|---|
| What is the smallest direct solve? | `example_basic_solve` |
| My matrix is already CSR or CSC. | `example_compressed_input` |
| I need analyze-once / factor-many. | `example_analysis` |
| I need an iterative solve. | `example_iterative` |
| I need symmetric eigenpairs. | `example_eigs` |
| I need SVD or low-rank behavior. | `example_svd_lowrank` |
| I need COLAMD or reorder guidance. | `example_colamd` |
| I need installed CMake consumption. | `examples/cmake_example/` |
| I need Matrix Market load/use. | `example_matrix_market` |

## Benchmark Handoff

Move to benchmarks after the API workflow is chosen and you need local
measurement:

- `bench_refactor` and `bench_refactor_csc` for repeated direct reuse.
- `bench_iterative_reuse` for iterative handles.
- `bench_eigs_reuse` for eigensolver handles.
- `bench_reorder` and `bench_colamd` for ordering comparisons.
- `make bench-canonical-report` for a threshold-free local snapshot of the
  maintained benchmark surface.

Benchmark rows are measurement artifacts. They do not replace examples,
headers, tests, or solver assumptions. Compare rows only when the machine,
compiler, backend selection, matrix corpus, build options, and thread settings
are recorded or intentionally held fixed.
