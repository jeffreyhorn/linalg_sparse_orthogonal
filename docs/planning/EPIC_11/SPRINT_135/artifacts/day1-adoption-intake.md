# Sprint 135 Day 1 - Adoption Intake

## Purpose

Day 1 establishes the Sprint 135 baseline for adoption-surface
simplification, documentation productization, compressed-first workflow
discoverability, artifact ownership, and claim boundaries.

## Project Plan Mapping

| Item | Project-plan scope | Sprint 135 day owners |
| --- | --- | --- |
| 1 | Adoption Surface Audit | Days 1-2 |
| 2 | Algorithm Doc Split Design | Days 3-4 |
| 3 | Algorithm Doc Split Batch | Days 4-6 |
| 4 | Compressed-First Cookbook | Days 7-9 |
| 5 | Benchmark/Report Index Docs | Days 10-11 |
| 6 | Link and Claim Validation | Days 12-13 |
| 7 | Closeout | Day 14 |

## Inherited Report and Package Baseline

| Surface | Inherited truth |
| --- | --- |
| Report indexes | Generated indexes are report traceability and freshness evidence, not broad correctness proof. |
| First generated index | The large-matrix guardrail `index.tsv` is accepted as the first generated index path without creating a normalized cross-report schema. |
| Coverage reports | Supplemental and tree-mutating; coverage percentage is not reviewed behavioral completeness. |
| Dead-code reports | Conservative report-completeness evidence, not zero-findings or removal-ready proof. |
| Benchmark outputs | Local measurement evidence, not portable performance or correctness guarantees. |
| Package shape | Static-first package contract remains maintained. |
| Shared-library support | Deferred non-claim. |
| Dynamic ABI compatibility | Deferred non-claim. |
| Package-manager support | Deferred non-claim. |
| Platform tiers | Linux reviewed static-first package-contract CI; macOS and Windows package install/export confidence remain supplemental. |

## Current Adoption Surface Inventory

| Surface | Current role | Sprint 135 question |
| --- | --- | --- |
| `README.md` | Front-door feature, build, install, CI, examples, and docs index. | Can the first-use path be shorter while preserving support-tier truth? |
| `INSTALL.md` | Static-first install, package-consumer, and platform setup guidance. | Can install guidance stay linked from adoption docs without becoming duplicated support history? |
| `docs/tutorial.md` | First-use tutorial and workflow introduction. | Should tutorial content become the primary guided path or delegate more to cookbook/reference docs? |
| `docs/solver_selection.md` | Solver-choice and decision guidance. | Can solver choice link into compressed-first cookbook paths without repeating algorithm history? |
| `docs/algorithm.md` | Algorithm reference mixed with current behavior and historical measurement context. | Which content becomes concise current reference, and which content moves to historical appendix? |
| `docs/matrix_market.md` | Matrix Market input and compressed representation guidance. | Can Matrix Market become an explicit compressed-first cookbook path? |
| `docs/maintainer_guide.md` | Maintainer history, support tiers, proof ownership, and residual queues. | Which maintainer/history links should remain out of first-use adoption flow? |
| `examples/README.md` | Maintained example index. | Can examples be grouped by adoption workflow family? |
| `benchmarks/README.md` | Benchmark commands and measurement entry point. | Can benchmark/report interpretation become concise and evidence-bounded? |

## Maintained Workflow Source Inventory

| Workflow family | Maintained sources | Sprint 135 adoption target |
| --- | --- | --- |
| Direct compressed-first solve | `examples/example_basic_solve.c`, `examples/example_compressed_input.c` | Concise direct-solver cookbook path with links to current solver guidance. |
| Iterative solve | `examples/example_iterative.c`, `examples/example_ic_minres.c`, `examples/example_matrix_free.c` | Concise iterative cookbook path with preconditioner and matrix-free links. |
| Matrix Market | `examples/example_matrix_market.c`, `docs/matrix_market.md` | Matrix Market input path that leads into compressed storage and solver selection. |
| SVD | `examples/example_svd_lowrank.c` | Low-rank SVD cookbook path linked to current algorithm/reference text. |
| Eigensolver | `examples/example_eigs.c` | Eigensolver cookbook path linked to current algorithm/reference text. |
| Benchmark and measurement | `benchmarks/README.md`, `benchmarks/bench_*.c` | Local measurement path linked to report-index interpretation without portable performance claims. |
| Installed consumer | `examples/cmake_example/` | Downstream consumer path linked from install docs rather than solver cookbook details. |

## First-Use Versus Maintainer Surface Classes

| Class | Candidate surfaces | Notes |
| --- | --- | --- |
| First-use guide | `README.md`, `docs/tutorial.md`, future cookbook sections, `examples/README.md` | Should answer what to read and run first. |
| Concise reference | `docs/solver_selection.md`, split current sections from `docs/algorithm.md`, `docs/matrix_market.md` | Should describe current behavior and limitations without historical measurement narrative. |
| Generated-report index | Sprint 131 report artifacts, large-matrix guardrail index, benchmark/report docs | Should describe source, freshness, and interpretation boundaries. |
| Maintainer history | `docs/maintainer_guide.md`, planning artifacts, historical measurement appendix | Should remain discoverable without interrupting adoption flow. |
| Historical measurement appendix | Candidate split target from `docs/algorithm.md` and benchmark/report history | Should preserve context while avoiding default first-use prominence. |

## Claim Fences

- First-use guidance may simplify paths, but it must not simplify away
  support-tier distinctions.
- Current algorithm reference may describe supported behavior, but it must not
  imply external library, backend, basis-vector, sign, orientation, or broad
  oracle parity.
- Historical measurement material may be moved or linked, but benchmark rows
  remain local measurements, not portable performance guarantees.
- Generated report indexes remain traceability and freshness evidence, not
  correctness, coverage-completeness, or release guarantees.
- Cookbook examples must link to maintained example sources instead of
  copying source-level implementation details into multiple docs.
- Install/package guidance must preserve static-first support and explicit
  shared-library, dynamic ABI, runtime-loader, and package-manager non-claims.
- Platform wording must preserve Sprint 134 tiers: Linux reviewed package
  contract, macOS supplemental package confidence, Windows supplemental
  install/downstream confidence, and staged Windows pthread/POSIX tests.

## Day 2 Handoff

Day 2 should perform the formal adoption surface audit:

- inspect README, tutorial, solver-selection, algorithm, Matrix Market,
  benchmark, install, maintainer, and example docs for duplication and
  adoption friction;
- classify major sections as first-use, concise reference, generated-report
  index, maintainer history, or historical measurement appendix candidate;
- record compressed-first discoverability gaps by direct, iterative, Matrix
  Market, SVD, eigensolver, and benchmark workflow family;
- identify link/path dependencies before any Day 3-6 algorithm-document
  movement.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 135 project-plan item has a day-level owner. | Complete | Working notes and this artifact map Items 1-7 to Days 1-14. |
| Prior report-index, package, ABI, and platform truth is preserved before documentation changes begin. | Complete | Inherited baseline and claim fences preserve Sprint 131 and Sprint 133-134 support boundaries. |
| First-use, reference, maintainer, historical, and generated-report surfaces are visible before simplification decisions begin. | Complete | Adoption inventory and surface-class table identify current documents and their candidate roles. |
