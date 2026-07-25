# Sprint 135 Day 7 - Compressed-First Cookbook Design

## Purpose

Design the compressed-first cookbook before editing the public adoption docs.
The goal is a concise first-use route for callers whose data already arrives as
CSR, CSC, or Matrix Market input, without turning the cookbook into another API
reference, benchmark manual, or maintainer history page.

## Current Surface Inventory

| Workflow | Current public entry points | Current docs/examples | Current gap |
|---|---|---|---|
| Direct one-shot from CSR/CSC | `sparse_create_from_csr(...)`, `sparse_create_from_csc(...)`, `sparse_from_csr(...)`, `sparse_from_csc(...)`, then LU/Cholesky/LDL^T/QR as appropriate | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `examples/example_compressed_input.c`, `examples/README.md` | The compressed constructor path is discoverable, but the direct-solver handoff is split across several pages. |
| Stable-pattern direct reuse | `sparse_analyze(...)`, `sparse_factor_numeric(...)`, `sparse_factor_solve(...)`, `sparse_refactor_numeric(...)` | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `examples/example_analysis.c`, `examples/README.md` | Reuse is well documented, but the cookbook should say when compressed-first callers still move from construction into the same analysis lifecycle. |
| Iterative solve | `sparse_solve_cg(...)`, `sparse_solve_gmres(...)`, `sparse_solve_minres(...)`, preconditioner callbacks, one-shot `BiCGSTAB` compatibility | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `examples/example_iterative.c`, `examples/example_ic_minres.c`, `examples/example_matrix_free.c` | Existing examples start from constructed matrices or Matrix Market input; the cookbook should make the CSR/CSC-to-iterative handoff explicit. |
| Matrix Market input | `sparse_load_mm(...)`, `sparse_save_mm(...)` | `docs/matrix_market.md`, `examples/example_matrix_market.c`, `README.md`, `docs/solver_selection.md` | Format support is clear, but cookbook adoption should treat loaded Matrix Market matrices as the same public matrix shell used by every solver family. |
| SVD and low-rank | `sparse_svd_compute(...)`, `sparse_svd_partial(...)`, `sparse_svd_lowrank(...)`, `sparse_svd_lowrank_sparse(...)`, `sparse_cond(...)` | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/algorithm.md`, `examples/example_svd_lowrank.c` | SVD guidance exists, but it is not presented as a short compressed-first route from loaded/imported sparse data into dense or sparse low-rank outputs. |
| Symmetric eigensolver | `sparse_eigs_sym(...)`, explicit eigensolver handle APIs for stable-dimension reuse | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/algorithm.md`, `examples/example_eigs.c` | The eigensolver example is strong, but the cookbook should state the symmetric-matrix precondition before sending compressed-first callers there. |
| Benchmarks and reports | `make bench-canonical-report`, `make performance-sentinels`, individual `bench_*` binaries | `benchmarks/README.md`, `README.md`, `docs/solver_selection.md`, `examples/README.md` | Benchmark docs are authoritative; the cookbook only needs a handoff from chosen workflow to local measurement, plus a no-portable-performance-claim caveat. |

## Target Cookbook Structure

Create `docs/cookbook.md` as a short task-oriented page with this outline:

1. `# Cookbook`
2. `## Start From Your Data`
   - CSR arrays
   - CSC arrays
   - Matrix Market files
   - small hand-written matrices as the exception, not the compressed-first
     target
3. `## Direct Solves From Compressed Input`
   - import CSR/CSC into the public matrix shell
   - choose LU, Cholesky, LDL^T, or QR by problem shape
   - copy before one-shot in-place factorization when the original matrix view
     must survive
   - move to `example_analysis` for stable-pattern repeated direct reuse
4. `## Iterative Solves From Compressed Input`
   - construct or load the public matrix shell first
   - choose CG, GMRES, MINRES, or BiCGSTAB by assumptions
   - choose IC(0), ILU(0), or ILUT by solver/matrix assumptions
   - keep matrix-free workflows separate because they do not start from CSR/CSC
     arrays
5. `## Matrix Market Load/Use`
   - load with `sparse_load_mm(...)`
   - handle parse/I/O errors through `sparse_err_t` and `sparse_errno()`
   - route the loaded matrix through direct, iterative, SVD, eigensolver, or
     benchmark workflows by problem shape
6. `## SVD and Low-Rank Workflows`
   - use the public matrix shell from imported/loaded sparse data
   - choose full SVD, partial SVD, condition, rank, pseudoinverse, dense
     low-rank, or sparse low-rank by output need
   - point to `example_svd_lowrank` and `sparse_svd.h`
7. `## Symmetric Eigensolver Workflows`
   - require symmetric input
   - start with `SPARSE_EIGS_BACKEND_AUTO`
   - reserve shift-invert, preconditioning, and explicit backends for advanced
     cases
   - point to `example_eigs` and `sparse_eigs.h`
8. `## Measure After Choosing the API Workflow`
   - use `benchmarks/README.md` as the benchmark authority
   - map API workflows to benchmark families
   - state benchmark rows are local measurement artifacts, not portable timing
     guarantees

The cookbook should not duplicate long signatures from headers. It should use
small code fragments only where they remove friction from the first handoff:

- CSR/CSC import skeleton
- Matrix Market load skeleton
- direct/iterative/SVD/eigensolver routing snippets no longer than necessary

## Link and Navigation Plan

Add cookbook links from:

- `README.md` documentation index and first-use workflow section
- `docs/tutorial.md` getting-started handoff
- `docs/solver_selection.md` front matter or example handoff
- `examples/README.md` start-here section

Do not make `benchmarks/README.md` depend on the cookbook. Benchmarks should
remain a measurement authority that adoption docs point into, not another
first-use navigation page.

## Day 8 Implementation Queue

Direct, iterative, and Matrix Market cookbook content:

1. Add `docs/cookbook.md` with the shared title, start-from-data section, and
   direct-solver route.
2. Add iterative-solver route, including preconditioner-family caveats and
   matrix-free separation.
3. Add Matrix Market load/use route and link back to `docs/matrix_market.md`.
4. Add initial inbound links from README, tutorial, solver-selection, and
   examples README.
5. Keep examples as runnable handoffs rather than copying full example source.

## Day 9 Implementation Queue

SVD, eigensolver, and benchmark cookbook content:

1. Add SVD/low-rank route with output ownership and dense/sparse output
   boundaries.
2. Add symmetric eigensolver route with symmetry, backend, shift-invert, and
   handle-reuse boundaries.
3. Add benchmark/report handoff, mapping workflows to benchmark groups while
   preserving `benchmarks/README.md` authority.
4. Recheck support-tier, package, ABI, and performance claims after navigation
   links are added.

## Claim Boundaries

The cookbook must preserve these fences:

- no package-manager availability claim
- no shared-library or dynamic-ABI guarantee
- no portable performance guarantee
- no state-of-the-art parity claim
- no nonsymmetric eigensolver claim
- no claim that benchmark reports are pass/fail timing gates, except the
  existing narrow `wall-check` lane documented elsewhere
- no claim that compressed constructors adopt caller arrays; they validate and
  copy
- no claim that repeated-run handles cover `BiCGSTAB` or block iterative
  workflows

## Validation Plan

Documentation-only Day 8/9 implementation should run:

- `git diff --check`
- `rg -n '[[:blank:]]$' README.md docs/cookbook.md docs/tutorial.md docs/solver_selection.md examples/README.md docs/matrix_market.md benchmarks/README.md`
- `test -f docs/cookbook.md && test -f examples/example_compressed_input.c && test -f examples/example_matrix_market.c && test -f examples/example_svd_lowrank.c && test -f examples/example_eigs.c && test -f benchmarks/README.md`
- focused link scan for `docs/cookbook.md`, `examples/README.md`,
  `docs/solver_selection.md`, `docs/matrix_market.md`, and
  `benchmarks/README.md`
- unsupported-claim scan for package, ABI, platform, and performance wording
- `git diff --name-only -- '*.c' '*.h'` to confirm no code-day quality gate is
  required unless cookbook implementation touches source or headers

## Completion Criteria

- every requested compressed-first workflow has a planned adoption route
- cookbook scope is separated from exhaustive API reference, maintainer
  history, and benchmark command ownership
- implementation is split cleanly between Day 8 and Day 9
- support, package, ABI, and performance claim boundaries are explicit before
  public docs are edited
