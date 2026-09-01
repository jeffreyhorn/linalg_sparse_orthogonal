# Cookbook

Use this cookbook when your matrix data already exists outside the library and
you want the shortest route from that data into a supported public workflow.
It complements the [solver-selection guide](solver_selection.md), the
[tutorial](tutorial.md), and the runnable [examples](../examples/README.md).

For install or downstream consumer setup, use [INSTALL.md](../INSTALL.md). For
current algorithm behavior, use [algorithm.md](algorithm.md); historical
measurement notes live in [algorithm_history.md](algorithm_history.md).
For exact declarations, option/result structs, ownership rules, and return-code
contracts after you choose a workflow, use [api_reference.md](api_reference.md)
and the public headers under [`include/`](../include/).

## First-Use Ladder

Use this ladder when you are moving from data to the first maintained example:

1. Run [`example_basic_solve`](../examples/example_basic_solve.c) when you want
   the smallest hand-written one-shot direct solve.
2. Run
   [`example_compressed_input`](../examples/example_compressed_input.c) when
   your coefficients already live in CSR or CSC arrays.
3. Use [solver_selection.md](solver_selection.md) when the problem is not a
   general square one-shot direct solve.
4. Inspect diagnostics from the same workflow that produced them:
   construction errors, direct residuals, iterative convergence/status, QR
   rank/residuals, or SVD rank/condition output. Use
   [solver_selection.md#diagnostics-handoff](solver_selection.md#diagnostics-handoff)
   before changing solver family, backend, preconditioner, tolerance, or
   benchmark settings.
5. Use [INSTALL.md](../INSTALL.md) and
   [`examples/cmake_example/`](../examples/cmake_example/) only when you need
   installed downstream consumption rather than a build-tree example.

The ladder is a routing aid. It does not widen QR, SVD, runtime, package,
platform, performance, or state-of-the-art claims beyond the maintained proof
named in the linked sections.

## Start From Your Data

| Your data starts as | First public step | Then choose |
|---|---|---|
| Caller-owned CSR arrays | `sparse_create_from_csr(...)` or `sparse_from_csr(...)` | Direct, iterative, SVD, or symmetric eigensolver workflow |
| Caller-owned CSC arrays | `sparse_create_from_csc(...)` or `sparse_from_csc(...)` | Direct, iterative, SVD, or symmetric eigensolver workflow |
| Matrix Market file | `sparse_load_mm(...)` | Same solver-selection path as any returned `SparseMatrix *` |
| Small hand-written matrix | `sparse_create(...)` plus `sparse_insert(...)` | Small examples, tests, or quick experiments |

CSR and CSC constructors validate and copy the compressed arrays into the
normal public matrix shell. The caller still owns the input arrays after
construction, and the returned matrix is freed with `sparse_free(...)`.

Use the `sparse_create_from_*` constructors when a `NULL` result is enough for
invalid input. Use `sparse_from_*` when the call site needs an explicit
`sparse_err_t` diagnostic.

After construction, choose the solver by problem shape rather than by storage
format. CSR, CSC, Matrix Market, and small hand-written matrices all become
normal public `SparseMatrix *` values before they enter direct, iterative, QR,
SVD, or eigensolver workflows.

## Direct Solves From Compressed Input

Start here when the problem is direct-solver shaped and the matrix already
arrives as CSR or CSC data.

1. Build the public matrix shell from the compressed arrays:

   ```c
   SparseMatrix *A = sparse_create_from_csr(&csr);
   if (!A) {
       /* handle invalid input or allocation failure */
   }
   ```

2. Choose the direct solver by problem shape:

   | Problem shape | First solver |
   |---|---|
   | General square system | LU |
   | Symmetric positive-definite system | Cholesky |
   | Symmetric indefinite system | LDL^T |
   | Rectangular, least-squares, or rank-sensitive system | QR |

   The maintained QR corpus proof covers the bounded Sprint 139/Sprint 150
   fixture family owned by `tests/test_qr_corpus.c`: the
   `qr_rank_deficient_6x4_nullspace_v1` seed, two rank-deficient rectangular
   fixtures, and three underdetermined minimum-norm fixtures. The selected
   comparison freshness gate also covers selected QR minimum-norm, compatible
   least-squares, and incompatible least-squares rows from
   `tests/corpus/manifests/selected_report_targets.tsv`. Use these as
   fixture-local QR confidence, not as broad QR, broad least-squares, or
   external-library parity.

3. If the one-shot factorization mutates its input and you still need the
   original matrix view, factor a fresh copy:

   ```c
   SparseMatrix *LU = sparse_copy(A);
   if (!LU) {
       sparse_free(A);
       return SPARSE_ERR_ALLOC;
   }

   sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
   if (err != SPARSE_OK) {
       sparse_free(LU);
       sparse_free(A);
       return err;
   }
   ```

4. Free both the factor copy and the original matrix shell when done:

   ```c
   sparse_free(LU);
   sparse_free(A);
   ```

Use [`examples/example_compressed_input.c`](../examples/example_compressed_input.c)
for the smallest runnable CSR/CSC import and one-shot direct solve. Use
[`examples/example_basic_solve.c`](../examples/example_basic_solve.c) for the
smallest hand-written direct solve.

Diagnostics for this first-use route are deliberately simple:

- `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)` return
  `NULL` for invalid input or allocation failure;
- `sparse_from_csr(...)` and `sparse_from_csc(...)` return explicit
  `sparse_err_t` values when the caller needs a diagnostic owner;
- direct factorization and solve calls return `sparse_err_t`;
- residual checks in examples are local confidence checks for the shown
  matrix, not broad numerical or performance proof.

When the sparsity pattern is stable across many value changes, move from
one-shot direct calls to the explicit repeated-run lifecycle:

1. `sparse_analyze(...)`
2. `sparse_factor_numeric(...)`
3. `sparse_factor_solve(...)`
4. `sparse_refactor_numeric(...)` before later same-pattern solves

Use [`examples/example_analysis.c`](../examples/example_analysis.c) for that
analyze-once / factor-many route. The lifecycle is for same-pattern reuse; it
is not an implicit structural rebuild path.

## Iterative Solves From Compressed Input

Iterative solvers use the same public matrix shell as direct solvers. If your
coefficients arrive as CSR or CSC arrays, import them first, then choose the
iterative family by matrix assumptions.

| Problem shape | First solver | Reuse support |
|---|---|---|
| Symmetric positive-definite | CG | Repeated-run handle supported |
| General unsymmetric | GMRES | Repeated-run handle supported |
| Symmetric indefinite | MINRES | Repeated-run handle supported |
| General nonsymmetric compatibility path | BiCGSTAB | One-shot only |

Use the input `x` vector as the initial guess. Pass a zeroed vector when there
is no useful prior guess.

Choose preconditioners by the same assumptions:

- IC(0) for SPD operators and CG/MINRES workflows
- ILU(0) or ILUT for general or indefinite workflows

Preconditioner setup routines expect an original matrix view with identity
permutations. If the matrix may already have been factored or reordered, build
the preconditioner from a fresh `sparse_copy()` of the original matrix.

Use [`examples/example_iterative.c`](../examples/example_iterative.c) for a
one-shot GMRES workflow with and without ILU(0). Use
[`examples/example_ic_minres.c`](../examples/example_ic_minres.c) for IC(0),
CG, and MINRES assumptions.

Matrix-free iterative workflows are separate: use
[`examples/example_matrix_free.c`](../examples/example_matrix_free.c) when the
operator is naturally a callback rather than CSR, CSC, or Matrix Market data.

## Matrix Market Load/Use

Use Matrix Market when the matrix arrives as a `.mtx` file:

```c
SparseMatrix *A = NULL;
sparse_err_t err = sparse_load_mm(&A, "matrix.mtx");
if (err != SPARSE_OK) {
    if (err == SPARSE_ERR_IO) {
        int saved_errno = sparse_errno();
        (void)saved_errno;
    }
    /* handle parse, I/O, allocation, or argument error */
}
```

On success, `A` is a normal caller-owned `SparseMatrix *`. Route it through
the same solver-selection rules as any other public matrix shell, then free it
with `sparse_free(A)`.

Use [`docs/matrix_market.md`](matrix_market.md) for supported Matrix Market
headers, duplicate-entry behavior, symmetric expansion, zero elision,
ownership, and errno details. Use
[`examples/example_matrix_market.c`](../examples/example_matrix_market.c) for
the smallest runnable load/use solve.

## SVD and Low-Rank Workflows

Use SVD APIs after CSR, CSC, or Matrix Market input has been converted into the
normal public matrix shell. Keep the original matrix view unfactored and
unreordered for SVD calls; if matrix state is uncertain, start from a fresh
`sparse_copy()` of the original coefficients before mutating another working
matrix elsewhere.

Choose the SVD path by output need:

| Need | Public route |
|---|---|
| All singular values, optionally vectors | `sparse_svd_compute(...)` |
| Largest `k` singular values | `sparse_svd_partial(...)` |
| Numerical rank | `sparse_svd_rank(...)` |
| 2-norm condition estimate | `sparse_cond(...)` |
| Dense pseudoinverse | `sparse_pinv(...)` |
| Dense rank-`k` approximation | `sparse_svd_lowrank(...)` |
| Sparse dropped low-rank approximation | `sparse_svd_lowrank_sparse(...)` |

Dense SVD outputs and low-rank buffers are caller-owned according to the public
SVD API. Sparse low-rank output is returned as a `SparseMatrix *` and is freed
with `sparse_free(...)`.

The maintained corpus includes fixture-local partial-SVD rows for
`partial_svd_clustered_repeated_diag8x6_k3_v1`,
`partial_svd_rankdef_diag6x4_k2_range_projector_v1`,
`partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and
`partial_svd_fail_closed_diag6_k2_v1`. Use them as regression proof for those
generated cases only: top-k values, projector/residual/orthogonality checks,
sparse low-rank shape/nnz/selected values/Frobenius behavior, tight-budget
fail-closed behavior, and recovery. They do not claim broad partial-SVD
correctness, raw singular-vector identity, broad sparse-output optimality,
external-library parity, performance, platform/package/ABI support, or
state-of-the-art behavior.

Use [`examples/example_svd_lowrank.c`](../examples/example_svd_lowrank.c) for
the smallest runnable SVD, rank, condition, and low-rank workflow. Use
[`sparse_svd.h`](../include/sparse_svd.h) for exact option and ownership
details.

## Symmetric Eigensolver Workflows

Use `sparse_eigs_sym(...)` only when the problem is symmetric. Imported CSR,
CSC, or Matrix Market input still becomes a normal public matrix shell first;
the eigensolver choice happens after that construction or load step.

Start with the default backend behavior:

```c
sparse_eigs_opts_t opts = {
    .which = SPARSE_EIGS_LARGEST,
    .tol = 1e-10,
    .compute_vectors = 1,
};
```

Leave `opts.backend` at its zero default unless profiling or workload-specific
control justifies an explicit backend request. Shift-invert,
preconditioning, and explicit repeated-run eigensolver handles are advanced
paths; use them when the matrix assumptions and repeated-dimension reuse model
are clear.

Use [`examples/example_eigs.c`](../examples/example_eigs.c) for the runnable
symmetric eigensolver workflow, including shift-invert and a preconditioned
LOBPCG case. Use [`sparse_eigs.h`](../include/sparse_eigs.h) for exact option,
result, backend, and handle details.

This cookbook does not describe a nonsymmetric eigensolver workflow.

## Measure After Choosing the API Workflow

Move to benchmarks after the API workflow is chosen and you need local
measurement. Keep benchmark rows separate from examples: examples teach usage,
tests own regression behavior, and benchmarks provide branch-local measurement
artifacts.

| Chosen workflow | Measurement handoff |
|---|---|
| One-shot direct solve or broad solver harness | `bench_main` |
| Stable-pattern direct reuse | `bench_refactor` or `bench_refactor_csc` |
| Iterative handle reuse | `bench_iterative_reuse` |
| Symmetric eigensolver handle reuse | `bench_eigs_reuse` |
| SVD behavior | `bench_svd` |
| Reordering or fill behavior | `bench_reorder`, `bench_colamd`, or `bench_fillin` |
| Maintained local snapshot | `make bench-canonical-report` |
| Local sentinel bundle | `make performance-sentinels` |
| Large-matrix guardrail lanes | `make large-matrix-guardrails` |

Use [`benchmarks/README.md`](../benchmarks/README.md) for command syntax, CSV
fields, generated report artifacts, and measurement interpretation. Treat
benchmark output as local evidence tied to the current machine, compiler,
backend selection, matrix corpus, build options, and thread settings.

Generated report directories include index and manifest-style artifacts for
navigation:

- `build/bench-reports/canonical/index.tsv` plus `manifest.txt`
- `build/bench-reports/sentinels/sentinels.tsv` plus `manifest.txt`
- `build/bench-reports/large-matrix-guardrails/index.tsv` plus
  `manifest.txt`

In the sentinel bundle, `S5` is the existing local hard `wall-check` gate.
`S2` Cholesky CSC and `S3` LDLT KKT rows are threshold-free local backend
context only; read them with the recorded compiler, platform, build mode,
backend request/selection/fallback, and thread settings.

Read indexes as artifact maps and freshness context. Regenerate reports when
you need current evidence; do not edit generated rows by hand to make an old
artifact look current.

For a cross-family maintainer view, run:

```sh
python3 scripts/normalize_report_index.py --check
python3 scripts/normalize_report_index.py --check-freshness
```

Treat normalized rows as discovery and freshness diagnostics. They preserve
claim boundaries; they are not broad performance, coverage, package, or
platform evidence by themselves.

## Next Steps

After choosing the API workflow:

- use [solver_selection.md](solver_selection.md) for the compact decision tree
- use [tutorial.md](tutorial.md) for the fuller walkthrough
- use [examples/README.md](../examples/README.md) for runnable examples
- use [benchmarks/README.md](../benchmarks/README.md) only after you need
  local measurement rather than another API example
