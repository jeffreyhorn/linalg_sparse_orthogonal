# Solver Selection Guide

Use this guide when you know the shape of your sparse problem and need to
choose the smallest supported public workflow. Start from the way your matrix
arrives, then choose the solver family whose assumptions match the problem.

For compressed-first CSR, CSC, and Matrix Market adoption paths, use the
[cookbook](cookbook.md). For runnable examples, use
[`examples/README.md`](../examples/README.md). For API details, use the public
headers under [`include/`](../include/). For install support, use
[`INSTALL.md`](../INSTALL.md). For local performance measurement and generated
report indexes, use [`benchmarks/README.md`](../benchmarks/README.md).

## First-Use Solver Route

Use this short route before reading the family-specific detail below:

1. **Confirm the matrix entry point.** If the data is already CSR, CSC, or
   Matrix Market, build the public `SparseMatrix *` through the cookbook path
   instead of inserting entries one by one.
2. **Choose by mathematical shape.** Use LU for a general square system,
   Cholesky for symmetric positive-definite input, LDLT for symmetric
   indefinite input, QR for rectangular/rank-sensitive least-squares, iterative
   solvers when direct cost or memory is the issue, symmetric eigensolvers for
   eigenpairs, and SVD APIs for rank/condition/low-rank questions.
3. **Run one maintained example.** Start with `example_basic_solve` or
   `example_compressed_input`, then branch to the family-specific example named
   in [Example Handoff](#example-handoff).
4. **Inspect diagnostics locally.** Use
   [Diagnostics Handoff](#diagnostics-handoff) before changing backends,
   preconditioners, tolerances, or benchmark settings.
5. **Escalate only after the first solve is understood.** Runtime/backend
   controls and benchmarks are advanced tools for an already chosen workflow;
   they do not create portable performance, package, platform, or
   state-of-the-art claims.

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

## Diagnostics Handoff

Diagnostics belong to the workflow that produced them. Start with the smallest
signal that can explain the local result before escalating to advanced
controls:

| Workflow | First diagnostic to inspect | Escalate only when |
|---|---|---|
| CSR/CSC construction | `NULL` constructor result or explicit `sparse_err_t` from `sparse_from_*`. | The input arrays are valid and copied, but later solver behavior is still unclear. |
| Matrix Market input | `sparse_errno()` after `sparse_load_mm(...)` failure. | File parsing succeeds and solver choice remains the issue. |
| One-shot direct solve | Factorization/solve return code and a problem-local residual. | The matrix assumptions match the solver and residual behavior still needs investigation. |
| Repeated direct lifecycle | Analyze/factor/refactor return codes, same-pattern invariant, and solve residuals. | The sparsity pattern is stable and backend/reordering policy is the real question. |
| Iterative solve | Convergence status, residual norm/history, iteration count, stagnation, and breakdown fields. | The solver/preconditioner assumptions match the system and tuning is needed. |
| QR | Rank, residual, nullity/nullspace output from QR APIs or examples. | You are still inside the bounded QR workflow described in [QR Evidence Boundary](#qr-evidence-boundary). |
| SVD or partial SVD | Rank, condition, triplet residuals, convergence status, and fail-closed status. | You are still inside the bounded SVD workflow described in [SVD and Low-Rank Workflows](#svd-and-low-rank-workflows). |
| Eigensolver | Ritz residual, convergence count, selected backend, peak basis size, and shift-invert/preconditioner status. | The problem is symmetric and backend or preconditioner selection is now the target. |
| Benchmarks or reports | Matrix corpus, compiler, backend, thread settings, generated index, and manifest context. | You are measuring local behavior, not trying to prove portable runtime claims. |

Use [examples/README.md#diagnostics-handoff](../examples/README.md#diagnostics-handoff)
for the runnable-example view of the same rule. Use the public headers when
you need exact return-code, ownership, and result-struct semantics.

## Advanced-Control Escalation

Leave defaults in place for the first successful solve:

- one-shot direct solvers first, before repeated-run direct lifecycle;
- `SPARSE_EIGS_BACKEND_AUTO` first, before explicit Lanczos,
  thick-restart, or LOBPCG backend selection;
- zero-initialized runtime/backend option structs first, before typed
  backend or analysis/reordering overrides;
- no benchmark interpretation until the API workflow is already chosen.

Escalate to typed runtime/backend controls only when the local diagnostic
surface says the default path is not the right fit. Environment variables,
benchmark flags, report indexes, and sentinel rows are maintainer or
measurement controls; they are not public ABI promises, package guarantees,
platform parity claims, or portable performance claims.

## Direct Solvers

| Problem | Use | Notes |
|---|---|---|
| General square matrix | LU | Use a fresh matrix or `sparse_copy(...)` if you need the original later. |
| Symmetric positive-definite matrix | Cholesky | Non-SPD inputs report an error; do not use Cholesky as a general fallback. |
| Symmetric indefinite matrix | LDLT | Use when symmetry is part of the problem model, such as KKT-style systems. |
| Rectangular or rank-sensitive least-squares | QR | Use QR-specific APIs for rectangular, least-squares, minimum-norm, and rank-sensitive workflows. One maintained corpus lane now proves fixture-local rank `3`, nullity `1`, and nullspace residual behavior for `qr_rank_deficient_6x4_nullspace_v1`; it is not broad QR or external-library parity. |

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

## QR Evidence Boundary

The maintained QR corpus proof for
`qr_rank_deficient_6x4_nullspace_v1` is owned by
[`tests/test_qr_corpus.c`](../tests/test_qr_corpus.c) and the opt-in local
oracle command `python3 scripts/run_corpus_oracle.py --include-solver-qr`.
It supports only the selected fixture-local rank/nullity/nullspace residual
claim. It does not claim raw QR basis parity, global rank-threshold policy,
broad rank-deficient solve, SuiteSparse, LAPACK, NumPy, SciPy, platform,
performance, or state-of-the-art parity.

## SVD and Low-Rank Workflows

Use SVD APIs when you need singular values, numerical rank, condition
estimates, pseudoinverse behavior, or low-rank approximations. Treat dense
outputs and low-rank buffers according to the ownership rules of the public
SVD APIs; do not depend on private dense workspaces.

For partial SVD, the maintained Sprint 140 corpus proof is intentionally
fixture-local: `partial_svd_clustered_repeated_diag8x6_k3_v1` checks generated
8x6 clustered/repeated top-3 singular values, left/right subspace projectors,
triplet residuals, orthogonality, default-budget success, and tight-budget
fail-closed behavior through `tests/test_svd_partial_corpus.c` and
`python3 scripts/run_corpus_oracle.py --include-partial-svd`. Do not use that
lane as broad repeated-spectrum, raw vector identity, external-library parity,
platform, performance, package, ABI, or state-of-the-art evidence.

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
