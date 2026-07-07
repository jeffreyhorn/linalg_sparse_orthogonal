# Day 1 User Journey Inventory

## Purpose

Day 1 establishes the Sprint 111 adoption map before changing user-facing
documentation or examples. The main risk is mixing proof-owner and planning
language into adoption surfaces, or writing guidance that implies public Matrix
I/O, Matrix builder, benchmark, or solver contracts that the library does not
actually expose. This artifact records the current documentation surfaces,
first-time user entry points, maintainer-only surfaces, day-level ownership,
dependency order, and validation expectations for the sprint.

## Source Inputs

- `docs/planning/EPIC_10/PROJECT_PLAN.md`, Sprint 111 section.
- `docs/planning/EPIC_10/SPRINT_111/PLAN.md`.
- `README.md`.
- `INSTALL.md`.
- `docs/tutorial.md`.
- `docs/matrix_market.md`.
- `docs/algorithm.md`.
- `docs/maintainer_guide.md`.
- `examples/README.md`.
- `examples/*.c` and `examples/cmake_example/`.
- `benchmarks/README.md` and benchmark drivers.
- Public headers under `include/`.
- Sprint 101-110 planning artifacts that provide evidence but should not be
  first-time adoption material.

## Adoption Surface Inventory

| Surface | Files | User-Facing Role | Day 1 Finding |
|---|---|---|---|
| README front door | `README.md` | First successful build/solve path, capability overview, workflow routing. | Already has a "Start Here" and "Choose a Workflow" shape; Sprint 111 should keep this concise and route deeper material out. |
| Install documentation | `INSTALL.md` | Platform install, downstream consumer, package-manager, and CMake detail. | Correct place for consumer setup; not the main solver-selection guide. |
| Tutorial | `docs/tutorial.md` | Deeper repeated-run and API walkthrough. | Useful as the second-level workflow guide after quick examples. |
| Matrix Market documentation | `docs/matrix_market.md` | Matrix Market load/save behavior and supported file formats. | Needs Sprint 111 tightening around ownership, duplicate entries, zero/default behavior, pattern/symmetric handling, errno, runtime, and no public Matrix I/O module claim. |
| Algorithm documentation | `docs/algorithm.md` | Algorithmic and implementation explanation. | Reference material, not a first-time adoption path. |
| Maintainer guide | `docs/maintainer_guide.md` | Quality policy, reviewed lanes, CI interpretation, maintainer workflow. | Maintainer-only for Sprint 111 audience purposes. |
| Examples index | `examples/README.md` | Compact map from README to runnable public examples. | Strong adoption surface; should align with the new solver-selection guide and compressed-first examples. |
| Example sources | `examples/example_*.c`, `examples/cmake_example/` | Copyable public API usage. | Primary place to make CSR/CSC, solver, Matrix Market, eigensolver, SVD, and downstream-consumer usage concrete. |
| Benchmark documentation | `benchmarks/README.md` | Benchmark command and output interpretation. | Needs responsible local-environment caveats and clearer relationship to solver-selection guidance. |
| Public headers | `include/*.h` | API names, option structures, ownership, errors, and lifecycle contracts. | Source of truth for public semantics; Day 2 must audit guide/example claims against these files. |
| Generated API reference | Doxygen output from `make docs` | Generated reference for public declarations. | Useful after adoption basics; not the first user route. |
| Planning artifacts | `docs/planning/EPIC_*` | Evidence, sprint plans, retrospectives, residual debt, proof-owner records. | Maintainer evidence only; summarize outcomes elsewhere instead of routing users here. |

## First-Time User Entry Point Map

| User Need | Primary Entry Point | Secondary Entry Point | Sprint 111 Owner |
|---|---|---|---|
| Build the project locally | `README.md` Building | `INSTALL.md` | Day 2 audit, Day 10 coherence |
| Install or use from a downstream CMake project | `INSTALL.md` | `examples/cmake_example/` | Day 2 audit, Day 10 coherence |
| Get one successful solve | `README.md` Quick Start, `examples/example_basic_solve.c` | `examples/README.md` | Day 5 audit, Day 7 examples |
| Start from CSR/CSC arrays | `examples/example_compressed_input.c` | `include/sparse_csr.h`, `include/sparse_matrix.h` | Day 5 audit, Day 6 examples |
| Load Matrix Market files | `docs/matrix_market.md` | future Matrix Market example/reference in examples docs | Day 8 examples, Day 9 docs |
| Choose LU/Cholesky/LDLT/QR | future solver-selection guide | `README.md`, direct solver headers | Day 3 outline, Day 4 guide |
| Reuse analysis/factors for stable patterns | `examples/example_analysis.c` | `docs/tutorial.md`, `include/sparse_analysis.h` | Day 4 guide, Day 7 examples |
| Choose iterative solvers | future solver-selection guide | `examples/example_iterative.c`, `include/sparse_iterative.h` | Day 4 guide, Day 7 examples |
| Use eigensolvers | `examples/example_eigs.c` | `include/sparse_eigs.h`, benchmark docs for measurement | Day 4 guide, Day 8 examples |
| Use SVD/low-rank workflows | `examples/example_svd_lowrank.c` | `include/sparse_svd.h` | Day 4 guide, Day 8 examples |
| Use reordering/fill workflows | `examples/example_colamd.c`, `examples/example_analysis.c` | `include/sparse_reorder.h`, `benchmarks/README.md` | Day 4 guide, Day 7 examples |
| Interpret benchmark output | `benchmarks/README.md` | benchmark drivers and canonical reports | Day 11 benchmark docs |

## Public Header Reference Map

| Header | Adoption Role |
|---|---|
| `include/sparse_matrix.h` | Core matrix lifecycle, insertion, Matrix Market load/save declarations, errors, and matrix state. |
| `include/sparse_csr.h` | CSR/CSC construction, export, and compressed-first support. |
| `include/sparse_lu.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_qr.h` | Direct solver entry points and option contracts. |
| `include/sparse_analysis.h` | Analyze/factor/solve/refactor lifecycle and typed analysis options. |
| `include/sparse_iterative.h` | CG, GMRES, MINRES, BiCGSTAB, handle, diagnostics, and convergence options. |
| `include/sparse_ilu.h`, `include/sparse_ic.h` | Preconditioner construction and lifecycle. |
| `include/sparse_eigs.h` | Symmetric eigensolver options, backend selection, handles, and result semantics. |
| `include/sparse_svd.h`, `include/sparse_bidiag.h` | SVD, low-rank, pseudoinverse, rank, and decomposition support. |
| `include/sparse_reorder.h` | RCM, AMD, ND, COLAMD, and ordering controls. |
| `include/sparse_dense.h`, `include/sparse_vector.h`, `include/sparse_types.h` | Supporting dense/vector/type contracts used by examples and advanced workflows. |

## Maintainer-Only Proof Surface List

These surfaces should remain available for project governance and evidence, but
they should not be the primary path for a user trying to build, choose a
solver, or write a small example:

- Epic project plans and sprint plans under `docs/planning/EPIC_*`.
- Sprint working notes and artifacts under
  `docs/planning/EPIC_*/SPRINT_*/`.
- Epic and sprint retrospectives.
- Review files under `docs/planning/EPIC_*/reviews/`.
- Residual deferred-debt queues.
- Source-boundary and private-owner decision records.
- Proof-owner cleanup artifacts for giant tests.
- CI reviewed-count, drift, and platform-scope notes.
- Benchmark evidence artifacts that are not written as user interpretation
  docs.

Sprint 111 may summarize these records in user docs only when the summary is
needed to explain a supported workflow, a documented caveat, or a responsible
benchmark interpretation.

## Initial Workstream Ownership

| Project Plan Item | Initial Owner Surface | Planned Days |
|---|---|---|
| Item 1: User Journey Audit | README, install docs, tutorial, examples, Matrix Market docs, benchmark docs, public headers. | Days 1-2 |
| Item 2: Solver Selection Guide | New or existing docs guide, README links, examples links, public solver headers. | Days 3-4 |
| Item 3: Compressed-First Example Batch | `examples/README.md`, `examples/example_compressed_input.c`, direct/iterative/eigs/SVD examples. | Days 5-8 |
| Item 4: Matrix Market Behavior and Ownership Docs | `docs/matrix_market.md`, README/examples references, `include/sparse_matrix.h` wording if needed. | Days 8-10 |
| Item 5: Benchmark Interpretation Docs | `benchmarks/README.md`, benchmark command references, solver guide caveats. | Day 11 |
| Item 6: Maintainer/User Split Cleanup | README, examples docs, tutorial, benchmark docs, maintainer guide routing. | Days 10 and 12 |
| Item 7: Validation and Closeout | Touched docs/examples/headers, Sprint 111 artifacts, working notes, residual queue. | Days 13-14 |

## Dependency-Ordered Queue

| Order | Work | Reason |
|---:|---|---|
| 1 | User journey inventory | Defines the adoption surfaces and audience boundary. |
| 2 | Documentation gap audit | Finds stale claims before guide or example edits. |
| 3 | Solver-selection outline | Establishes supported decisions before prose and examples. |
| 4 | Solver-selection guide draft | Gives examples and docs a shared target. |
| 5 | Example audit | Selects compressed-first workflows before source edits. |
| 6 | Construction examples | Makes the compressed-first path concrete. |
| 7 | Solver workflow examples | Aligns direct, iterative, reuse, and reorder/fill examples with the guide. |
| 8 | Advanced and Matrix Market examples | Adds higher-level workflows after basic examples are stable. |
| 9 | Matrix Market behavior docs | Documents precise behavior after example needs are visible. |
| 10 | Header/tutorial coherence | Aligns public wording after guide/docs wording stabilizes. |
| 11 | Benchmark interpretation docs | Connects benchmark caveats to the final solver workflow language. |
| 12 | Maintainer/user split cleanup | Removes remaining audience mixing before validation. |
| 13 | Integrated validation | Checks the full documentation/example surface. |
| 14 | Closeout and residual handoff | Records final state after validation. |

## Validation Expectations

| Scenario | Required Validation |
|---|---|
| Documentation-only changes | `git diff --check` and trailing-whitespace scan over touched docs. |
| Example source changes | Focused example build or `make examples`, plus `git diff --check`. |
| Public header changes | Public API/install-header review and checks appropriate to any code-adjacent changes. |
| Benchmark documentation changes | Claim review against current benchmark commands and artifacts, plus `git diff --check`. |
| Mixed documentation and examples | Strongest applicable check from the touched file set. |
| Implementation `.c` or `.h` changes | Focused tests plus `make format && make lint && make test`. |

## Day 1 Completion Criteria Status

- Every Sprint 111 project-plan item has initial documentation or example
  ownership.
- User-facing entry points for build, matrix creation, Matrix Market loading,
  direct solve, iterative solve, eigensolve, SVD, reordering, and benchmarking
  are mapped.
- Maintainer-only proof surfaces are distinguished from adoption surfaces.
- Downstream days can proceed from a known documentation layout instead of
  rediscovering it.
