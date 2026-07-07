# Day 3 Solver Selection Guide Outline

## Purpose

Day 3 defines the structure and contract boundaries for the Sprint 111
solver-selection guide before writing user-facing guide prose. The guide should
help users choose a public workflow from matrix format, problem structure,
reuse needs, and measurement needs. It must not become a maintainer proof
document, a benchmark claim surface, or a promise of unsupported public
subsystems.

## Source Inputs

- Day 1 user journey inventory.
- Day 2 documentation gap audit.
- `README.md` workflow and capability sections.
- `docs/tutorial.md` workflow walkthrough.
- `examples/README.md` and representative example programs.
- Public headers under `include/`, especially:
  - `include/sparse_matrix.h`
  - `include/sparse_csr.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_qr.h`
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_ilu.h`
  - `include/sparse_ic.h`
  - `include/sparse_eigs.h`
  - `include/sparse_svd.h`
  - `include/sparse_reorder.h`
- `benchmarks/README.md` for measurement caveats only.

## Guide Audience

The guide is for users who know what kind of sparse problem they have and need
to choose a supported public path:

- which matrix input path to use;
- which solver family to try first;
- when to use one-shot APIs versus explicit reusable lifecycles;
- when preconditioners or reordering are relevant;
- when examples teach usage and benchmarks measure local behavior.

The guide is not for maintainers deciding proof ownership, source boundaries,
reviewed CI counts, or deferred debt.

## Scope

The guide should cover:

- matrix format and ownership selection;
- one-shot direct solves;
- repeated direct solve lifecycle;
- iterative solver selection;
- preconditioner expectations;
- eigensolver and SVD workflow selection;
- reorder/fill workflow selection;
- example and benchmark handoff.

The guide should link to detailed docs instead of duplicating long reference
material.

## Non-Goals

- Do not claim a public Matrix I/O module.
- Do not claim a public Matrix builder API.
- Do not claim direct CSR/CSC public solve APIs.
- Do not claim universal performance portability.
- Do not claim broad external-oracle coverage for every solver family.
- Do not route first-time users into sprint plans, proof-owner artifacts, or
  residual-debt records.
- Do not describe private source ownership as a user-facing feature.

## Proposed Guide Structure

1. **Start From Your Matrix**
   - Caller already has CSR or CSC arrays.
   - Caller has a Matrix Market file.
   - Caller has a small hand-written matrix.
   - Caller needs dense-style output from SVD or low-rank workflows.
2. **Choose the Smallest Solve Workflow**
   - One-shot direct solve.
   - Repeated direct lifecycle.
   - One-shot iterative solve.
   - Repeated iterative handle.
   - Eigensolver or SVD workflow.
3. **Direct Solver Selection**
   - LU for general square systems.
   - Cholesky for SPD systems.
   - LDLT for symmetric indefinite systems.
   - QR for rectangular, least-squares, and rank-sensitive systems.
   - Reorder/fill guidance.
   - Fallbacks and failure interpretation.
4. **Iterative Solver Selection**
   - CG for SPD systems.
   - GMRES for general unsymmetric systems.
   - MINRES for symmetric indefinite systems.
   - BiCGSTAB for one-shot compatibility on nonsymmetric systems.
   - Preconditioners and convergence diagnostics.
   - Handle reuse boundaries.
5. **Eigensolver and SVD Selection**
   - Symmetric eigensolver route.
   - Backend selection at a user level.
   - SVD, rank, condition, pseudoinverse, and low-rank route.
   - Measurement and evidence caveats.
6. **Examples and Benchmarks**
   - Which example to start from.
   - When to move from examples to benchmarks.
   - What benchmark rows do and do not prove.

## Matrix-Format Decision Tree

| User Starting Point | Recommended Public Path | Ownership Notes | Follow-Up |
|---|---|---|---|
| CSR arrays | `sparse_create_from_csr(...)` for simple construction, or `sparse_from_csr(...)` for diagnostics. | Input arrays stay caller-owned; returned matrix is freed with `sparse_free(...)`. | `example_compressed_input`, direct or iterative solver guide. |
| CSC arrays | `sparse_create_from_csc(...)` or `sparse_from_csc(...)`. | Input arrays stay caller-owned; returned matrix is independent. | Direct solver guide; export/free with CSC helpers when needed. |
| Matrix Market file | `sparse_load_mm(...)`. | Loaded matrix is caller-owned and freed with `sparse_free(...)`; I/O errno is available through `sparse_errno()`. | Matrix Market docs and future Matrix Market example. |
| Small hand-written matrix | `sparse_create(...)` plus `sparse_insert(...)`. | Duplicate insertions overwrite through the public matrix semantics; zeros are not useful storage. | `example_basic_solve`, tutorial. |
| Existing public matrix shell | Use public copy/export/transpose/solve APIs. | Factorization may mutate working copies; use `sparse_copy(...)` when the original view must remain available. | Direct or iterative guide. |
| Dense-adjacent output need | Use public SVD, dense helper, or low-rank APIs as documented. | Do not depend on private dense workspaces or internal storage. | SVD guide and `example_svd_lowrank`. |

## Direct Solver Decision Notes

| Problem Shape | First Public Choice | When to Use | Caveats |
|---|---|---|---|
| General square system | LU | One-shot solve where no symmetry or definiteness contract is available. | One-shot LU factorization mutates the factor matrix; use a copy if the original is needed. |
| Symmetric positive-definite system | Cholesky | SPD systems where Cholesky is the natural model. | Non-SPD inputs report the appropriate error; do not use Cholesky as a general fallback. |
| Symmetric indefinite system | LDLT | KKT-style or indefinite symmetric systems. | Keep symmetry requirement explicit. |
| Rectangular or rank-sensitive least-squares | QR | Overdetermined, underdetermined, least-squares, minimum-norm, and rank-sensitive workflows. | Use QR-specific APIs rather than pretending LU/Cholesky cover rectangular systems. |
| Same sparsity pattern, many value changes | `sparse_analyze(...)`, `sparse_factor_numeric(...)`, `sparse_factor_solve(...)`, `sparse_refactor_numeric(...)`. | Reuse symbolic analysis and factor lifecycle objects across same-pattern solves. | Same-pattern requirements should be explicit; this is not a hidden rebuild path. |

Fallback guidance:

- If a direct solver rejects the problem because its structural assumptions are
  wrong, choose a solver whose assumptions match the matrix instead of forcing
  the same path.
- If repeated direct lifecycle setup is more complexity than the workload
  needs, stay on one-shot APIs.
- If direct solve cost or memory is the problem, consider iterative workflows
  with suitable preconditioning and diagnostics.

## Reorder and Fill Guidance

| Need | Public Path | Guide Boundary |
|---|---|---|
| Symmetric fill reduction | RCM, AMD, or ND as exposed through reorder or analysis options. | Explain as fill/work reduction, not correctness. |
| Unsymmetric or QR column ordering | COLAMD. | Distinguish column-only ordering from symmetric permutations. |
| Repeated direct lifecycle reordering | `sparse_analysis_opts_t` reorder settings. | Keep typed options as the guide path; legacy environment variables are not the adoption default. |
| Measurement of reorder choices | `bench_reorder`, `bench_colamd`, relevant benchmark docs. | Benchmarks measure local behavior; they do not establish portable guarantees. |

## Iterative Solver Decision Notes

| Problem Shape | First Public Choice | Reuse Support | Caveats |
|---|---|---|---|
| SPD system | CG | Handle reuse supported. | Matrix and preconditioner should satisfy SPD expectations. |
| General unsymmetric system | GMRES | Handle reuse supported. | Restart, tolerance, and memory settings matter. |
| Symmetric indefinite system | MINRES | Handle reuse supported. | Requires symmetry; preconditioner expectations are stricter than generic ILU use. |
| General nonsymmetric compatibility path | BiCGSTAB | One-shot only. | Use when it is the intended one-shot method; do not document handle reuse. |

Preconditioner wording:

- ILU and ILUT are acceleration tools for nonsymmetric/general workflows.
- IC is the symmetric/SPD-style preconditioner route.
- Preconditioners are problem-dependent; the guide should avoid promising
  convergence or speedup.
- Diagnostics such as iteration count, residual norm, stagnation, and
  breakdown should be part of the selection conversation.

## Eigensolver Guidance Boundaries

The guide should describe:

- `sparse_eigs_sym(...)` for symmetric sparse eigensolver workflows.
- `SPARSE_EIGS_BACKEND_AUTO` as the default user choice.
- Explicit backends as profiling or workload-specific overrides.
- Repeated-run eigensolver handles for stable-dimension reuse.
- Shift-invert and preconditioning as advanced paths with solver-specific
  requirements.

The guide should not claim:

- nonsymmetric eigensolver support;
- universal state-of-the-art parity;
- external ecosystem parity beyond measured local evidence;
- that benchmark examples prove all production workloads.

## SVD Guidance Boundaries

The guide should describe:

- SVD for singular values, rank, condition estimation, pseudoinverse, and
  low-rank approximation.
- `example_svd_lowrank` as the first copyable workflow.
- Dense-style output ownership and cleanup at the public API level.
- Partial SVD and low-rank workflows as available public features.

The guide should avoid:

- claiming complete external dense oracle coverage;
- implying private dense workspaces are part of the public API;
- turning benchmark timing into a correctness guarantee.

## Example Handoff Map

| User Question | Example |
|---|---|
| What is the smallest direct solve? | `example_basic_solve` |
| My matrix is already CSR/CSC. | `example_compressed_input` |
| I need analyze-once/factor-many. | `example_analysis` |
| I need an iterative solve. | `example_iterative` |
| I need symmetric eigenpairs. | `example_eigs` |
| I need SVD/low-rank behavior. | `example_svd_lowrank` |
| I need COLAMD/reorder guidance. | `example_colamd` |
| I need installed CMake consumption. | `examples/cmake_example/` |
| I need Matrix Market load/use. | Day 8 should add or identify the dedicated route. |

## Benchmark Handoff Rules

- Examples teach public API usage.
- Benchmarks measure local workflow/performance behavior.
- Benchmark output is branch-local and configuration-sensitive.
- Benchmark rows should not be described as portable timing guarantees.
- The guide may point to `benchmarks/README.md` when the user needs command
  syntax, CSV schema, or measurement interpretation.

## Day 4 Draft Requirements

The Day 4 solver-selection guide should:

- be concise enough to link from README and examples;
- start with matrix input format;
- make compressed-first workflows the default when the user already has CSR or
  CSC arrays;
- keep Matrix Market wording to public load/save functions;
- keep solver-family assumptions explicit;
- preserve repeated-run handle boundaries;
- include example handoff links;
- include benchmark caveats without importing maintainer proof language.

## Completion Criteria Status

- The guide structure covers matrix format, direct solve, iterative solve,
  eigensolver, SVD, reorder/fill, reuse, examples, and benchmark handoff.
- Claims are bounded by public headers, shipped examples, and documented
  evidence.
- Compressed-first workflows are the default route for caller-owned CSR/CSC
  data.
- Unknown, private, future, or maintainer-only behavior is excluded from user
  guidance.
