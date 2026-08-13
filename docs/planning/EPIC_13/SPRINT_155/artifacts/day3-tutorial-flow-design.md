# Sprint 155 Day 3 Tutorial Flow Design

## Purpose

Day 3 converts the Day 2 audit into a target tutorial structure for Days 4
and 5. The design keeps `docs/tutorial.md` as a user-facing learning path:
it should help a reader move from a local build to a first solve, then into
data input, solver choice, diagnostics, install, and advanced controls without
becoming a maintainer-policy document or a full API reference.

## Design Principles

1. **README remains the short front door.** The tutorial should assume the
   reader has seen the README and wants a fuller learning path.
2. **Examples remain runnable anchors.** The tutorial should point to
   maintained examples before deep API detail.
3. **Data input comes before solver depth.** CSR, CSC, Matrix Market, and
   hand-written matrices should all route into a normal `SparseMatrix *`
   before solver-family detail.
4. **Solver choice is problem-shaped.** The tutorial should align with
   `docs/solver_selection.md`, not create a second divergent decision tree.
5. **Diagnostics stay workflow-local.** Return codes, residuals, convergence,
   rank, condition, Ritz residuals, and report rows should be interpreted only
   in the context that produced them.
6. **Install and package detail stays delegated.** The tutorial may link to
   `INSTALL.md`, but it should not restate the static-first contract,
   platform-tier table, shared-library deferral, or package-manager non-claims.
7. **API reference owns exact declarations.** The tutorial can include small
   code snippets, but exact option structs, result fields, ownership rules, and
   public declarations should be delegated to public headers and the API
   reference plan.
8. **Evidence wording remains bounded.** QR, partial-SVD, comparison, report,
   benchmark, package, platform, and runtime/backend claims must stay tied to
   their owner surfaces and support tiers.

## Target Tutorial Outline

### 1. Getting Started

Purpose: mirror the maintained first-use ladder without duplicating the README.

Proposed content:

1. Build locally.
2. Run `./build/example_basic_solve`.
3. Start from CSR, CSC, or Matrix Market data when applicable.
4. Choose a solver family by problem shape.
5. Inspect workflow-local diagnostics.
6. Install only when a downstream consumer needs it.
7. Move to advanced controls, benchmarks, reports, public headers, or API
   reference only after the first workflow works.

Owner links:

- `README.md#start-here`
- `examples/README.md#start-here`
- `docs/cookbook.md#first-use-ladder`

### 2. Documentation Map

Purpose: explain which surface owns which kind of detail.

Keep concise rows for:

- solver selection;
- data-first recipes;
- runnable examples;
- install/downstream consumers;
- benchmark/report interpretation;
- public headers and API reference;
- maintainer policy.

Avoid adding maintainer-policy prose inside the tutorial.

### 3. Build-Tree Setup And First Solve

Purpose: give a reader one concrete success path before API breadth.

Proposed flow:

```sh
make
make examples
./build/example_basic_solve
```

Mention `make examples-build` only as the compile-only route when someone
wants to confirm examples build without running them. Keep `make test` as a
validation command, not as the first learning step.

The first-solve text should point readers to:

- `examples/README.md#one-shot-direct-example_basic_solve`;
- `README.md#quick-start` when they want pasteable source.

### 4. Link Or Install

Purpose: distinguish local build-tree snippets from installed consumers.

Keep a short build-tree compile example:

```sh
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```

Then delegate:

- `INSTALL.md#start-here` for static-first install;
- `INSTALL.md#using-via-pkg-config` for Makefile-style installed consumers;
- `INSTALL.md#using-from-a-cmake-project` for CMake installed consumers.

Do not duplicate static-first package contract details here.

### 5. Start From Your Matrix

Purpose: move data-input routing ahead of solver detail.

Proposed matrix-source table:

| Starting Data | Tutorial Route | Owner Link |
| --- | --- | --- |
| small hand-written matrix | `sparse_create(...)` and `sparse_insert(...)` | tutorial snippet |
| caller-owned CSR | `sparse_create_from_csr(...)` or `sparse_from_csr(...)` | cookbook |
| caller-owned CSC | `sparse_create_from_csc(...)` or `sparse_from_csc(...)` | cookbook |
| Matrix Market file | `sparse_load_mm(...)` | `docs/matrix_market.md` |

The section should emphasize that CSR/CSC constructors validate and copy input
arrays, and the returned `SparseMatrix *` is freed with `sparse_free(...)`.

### 6. Choose The Solver Workflow

Purpose: provide a compact tutorial-level route into solver families.

Proposed solver table:

| Need | First Workflow | Runnable Anchor |
| --- | --- | --- |
| one general square solve | LU | `example_basic_solve` |
| SPD direct solve | Cholesky | tutorial snippet and solver-selection link |
| symmetric indefinite direct solve | LDLT | `example_ldlt` |
| rectangular/rank-sensitive solve | QR | `example_least_squares`, `example_minnorm` |
| large or memory-sensitive system | iterative solver | `example_iterative`, `example_ic_minres` |
| symmetric eigenpairs | `sparse_eigs_sym(...)` | `example_eigs` |
| rank/condition/low-rank questions | SVD APIs | `example_svd_lowrank`, `example_condition` |
| procedural operator | matrix-free iterative | `example_matrix_free` |

This table should point to `docs/solver_selection.md` for the full decision
tree and evidence boundaries.

### 7. Direct Solver Walkthrough

Purpose: keep the existing LU, Cholesky, LDLT, and QR tutorial content, but
make it match the problem-shaped routing.

Day 4 should:

- keep LU and Cholesky snippets concise;
- add or strengthen the LDLT handoff to `example_ldlt`;
- keep QR guidance tied to original/unfactored matrix state;
- mention `sparse_qr_solve_minnorm(...)` for underdetermined minimum-norm
  solves;
- avoid broad QR or external-comparison claims.

### 8. Iterative And Preconditioned Workflows

Purpose: keep CG and GMRES snippets while adding MINRES and IC(0) routing.

Day 5 should:

- rewrite "ILU preconditioning dramatically reduces iteration counts" as
  local, workload-dependent acceleration guidance;
- state that preconditioners must match solver assumptions;
- add `example_ic_minres` as the IC(0), CG, and MINRES handoff;
- keep BiCGSTAB and block iterative workflows as one-shot compatibility
  surfaces where mentioned.

### 9. SVD And Low-Rank Workflows

Purpose: keep the existing SVD tutorial snippets but refresh evidence
boundaries.

Day 5 should either:

- summarize the full Sprint 151 partial-SVD fixture set narrowly; or
- delegate the detailed evidence list to
  `docs/solver_selection.md#svd-and-low-rank-workflows`.

Preferred approach: delegate the detailed evidence list to solver-selection
and keep tutorial text focused on which SVD API to call.

### 10. Symmetric Eigensolver Workflows

Purpose: close the missing tutorial coverage for `sparse_eigs_sym(...)`.

Content should stay compact:

- start with `SPARSE_EIGS_BACKEND_AUTO`;
- use `example_eigs` as the runnable anchor;
- mention Ritz residual and selected backend as diagnostics;
- avoid nonsymmetric eigensolver or state-of-the-art claims.

### 11. Matrix-Free Interface

Purpose: keep the existing callback snippets but position matrix-free as an
advanced iterative path after standard matrix-shell workflows.

Owner links:

- `examples/README.md#matrix-free-iterative-example_matrix_free`;
- `docs/solver_selection.md#iterative-solvers`.

### 12. Diagnostics Handoff

Purpose: replace the current generic-only error section with a workflow-local
diagnostics ladder plus the existing return-code table.

Proposed diagnostics table:

| Workflow | First Diagnostic |
| --- | --- |
| CSR/CSC construction | `NULL` result or explicit `sparse_err_t` |
| Matrix Market input | `sparse_errno()` after `SPARSE_ERR_IO` |
| one-shot direct | factor/solve return code and local residual |
| repeated direct | analyze/factor/refactor return code and same-pattern invariant |
| iterative | convergence status, residual, iterations, stagnation, breakdown |
| QR | rank, residual, nullity/nullspace outputs |
| SVD/partial-SVD | rank, condition, triplet residuals, convergence/fail-closed status |
| eigensolver | Ritz residual, converged count, backend used, peak basis size |
| benchmarks/reports | matrix, compiler, backend, thread settings, generated index/manifest |

Owner links:

- `docs/solver_selection.md#diagnostics-handoff`;
- `examples/README.md#diagnostics-handoff`;
- `benchmarks/README.md#reading-benchmark-results`.

### 13. Advanced Controls, Reports, API Reference, And Install

Purpose: keep advanced topics discoverable without turning the tutorial into
maintainer docs.

Recommended text:

- runtime/backend controls: link to README and solver-selection;
- benchmarks/reports: link to benchmarks README and maintainer guide;
- exact declarations/options/result structs: link to public headers and API
  reference plan;
- install/downstream: link to `INSTALL.md`.

Do not include report freshness commands unless the section is explicitly
about maintainer evidence validation.

## Section-To-Source Mapping

| Target Tutorial Section | Primary Source | Secondary Source |
| --- | --- | --- |
| Getting Started | `README.md#start-here` | `examples/README.md#start-here` |
| Documentation Map | `docs/maintainer_guide.md#documentation-ownership-rules` | README adoption map |
| Build-Tree Setup | `README.md#building` | `examples/README.md#building` |
| Link Or Install | `INSTALL.md#start-here` | `INSTALL.md#maintained-install-contract` |
| Start From Your Matrix | `docs/cookbook.md#start-from-your-data` | `docs/matrix_market.md` |
| Choose The Solver Workflow | `docs/solver_selection.md#choose-the-smallest-workflow` | `examples/README.md#programs` |
| Direct Solver Walkthrough | `docs/solver_selection.md#direct-solvers` | public direct headers |
| Iterative And Preconditioned Workflows | `docs/solver_selection.md#iterative-solvers` | `examples/README.md#ic0-cg-and-minres-example_ic_minres` |
| SVD And Low-Rank Workflows | `docs/solver_selection.md#svd-and-low-rank-workflows` | `examples/README.md#svd-low-rank-example_svd_lowrank` |
| Symmetric Eigensolver Workflows | `docs/solver_selection.md#eigensolver-workflows` | `examples/README.md#one-shot-symmetric-eigensolver-example_eigs` |
| Matrix-Free Interface | `examples/README.md#matrix-free-iterative-example_matrix_free` | public iterative headers |
| Diagnostics Handoff | `docs/solver_selection.md#diagnostics-handoff` | `examples/README.md#diagnostics-handoff` |
| Advanced Controls And Reports | `benchmarks/README.md#reading-benchmark-results` | `docs/maintainer_guide.md#normalized-report-index-workflow` |
| API Reference | generated `docs/api/html/` surface | public headers under `include/` |

## Example And Cookbook Cross-Link Plan

| Tutorial Need | Link Target |
| --- | --- |
| first maintained solve | `examples/README.md#one-shot-direct-example_basic_solve` |
| compressed CSR/CSC input | `examples/README.md#compressed-input-example_compressed_input` and `docs/cookbook.md#start-from-your-data` |
| Matrix Market load/use | `examples/README.md#matrix-market-loaduse-example_matrix_market` and `docs/matrix_market.md` |
| repeated direct lifecycle | `examples/README.md#repeated-run-direct-example_analysis` |
| least-squares/minimum-norm QR | `examples/README.md#rectangular-least-squares-example_least_squares` and `examples/README.md#underdetermined-minimum-norm-example_minnorm` |
| IC(0), CG, and MINRES | `examples/README.md#ic0-cg-and-minres-example_ic_minres` |
| SVD and condition | `examples/README.md#svd-low-rank-example_svd_lowrank` and `examples/README.md#condition-number-example_condition` |
| eigensolver | `examples/README.md#one-shot-symmetric-eigensolver-example_eigs` |
| matrix-free | `examples/README.md#matrix-free-iterative-example_matrix_free` |
| installed CMake consumer | `examples/README.md#installed-consumer-example-examplescmake_example` and `INSTALL.md` |

## Advanced-Control Boundary Notes

- Tutorial should mention advanced controls only after the basic workflow
  works.
- Runtime/backend options are public workflow controls, not ABI, package,
  platform, or portable performance claims.
- Benchmarks and reports are local measurement/evidence surfaces, not
  portable performance or release proof.
- Report freshness commands belong in maintainer or advanced-report contexts,
  not in first-use tutorial flow.
- Public headers and API reference own exact declarations, options, result
  structs, and ownership/error contracts.
- Generated Doxygen output should not be treated as fresh release proof until
  Days 10-11 define the reference publication plan.

## Day 4 Tutorial Rewrite Checklist

Day 4 should edit `docs/tutorial.md` through the core path only:

1. Reframe the opening around the maintained first-use ladder.
2. Keep or revise the documentation map so it explains owner surfaces.
3. Add the build-tree first-solve route:
   `make`, `make examples`, `./build/example_basic_solve`.
4. Keep local build-tree link guidance and delegate install consumers to
   `INSTALL.md`.
5. Move data-input routing before deep solver details.
6. Add first-solve and data-first cross-links to examples and cookbook.
7. Add a compact solver workflow table.
8. Avoid changing public headers or source code during the tutorial rewrite.

## Day 5 Tutorial Rewrite Checklist

Day 5 should finish tutorial alignment:

1. Rewrite the preconditioning claim.
2. Refresh or delegate partial-SVD evidence.
3. Add MINRES/IC(0) and eigensolver handoffs.
4. Add workflow-local diagnostics.
5. Add concise advanced-control, API-reference, report, and install handoffs.
6. Recheck unsupported claim wording.

## Day 3 Completion Check

- Target tutorial outline exists.
- Section-to-source mapping exists.
- Example and cookbook cross-link plan exists.
- Advanced-control boundaries are explicit.
- Day 4 and Day 5 rewrite checklists are concrete and bounded.
