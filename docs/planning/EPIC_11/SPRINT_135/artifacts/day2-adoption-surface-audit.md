# Sprint 135 Day 2 - Adoption Surface Audit

## Purpose

Day 2 audits the current first-use, reference, example, benchmark, install,
and maintainer documentation for overlap, adoption friction, compressed-first
discoverability gaps, and link/path dependencies that must be preserved during
later simplification work.

## Audited Surfaces

| Surface | Current role | Assigned Sprint 135 role |
| --- | --- | --- |
| `README.md` | Front-door feature overview, workflow chooser, build/setup summary, quick-start code, API overview, performance notes, limitations, quality, install, and doc index. | First-use guide and top-level navigation hub. |
| `INSTALL.md` | Install prerequisites, Make/CMake build, static-first install contract, package-consumer use, platform notes, and install validation. | Install/support reference linked from adoption flow. |
| `docs/tutorial.md` | Guided first-use walkthrough from build/linking through matrix construction, direct solvers, iterative solvers, SVD, and matrix-free usage. | First-use guide; candidate to delegate more to cookbook and concise reference. |
| `docs/solver_selection.md` | Problem-shape solver choice, examples handoff, and benchmark handoff. | Concise adoption reference and workflow router. |
| `docs/algorithm.md` | Large current algorithm reference mixed with sprint-era history, measurement narratives, report interpretation, performance gates, and implementation decision detail. | Split candidate: concise current algorithm reference plus historical measurement appendix. |
| `docs/matrix_market.md` | Matrix Market feature support, unsupported features, format reference, examples, and SuiteSparse notes. | Concise Matrix Market reference plus cookbook link target. |
| `docs/maintainer_guide.md` | Maintainer policy, support ownership, historical snapshots, package/platform truth, benchmark governance, documentation ownership, and repo norms. | Maintainer/history surface and claim-boundary authority. |
| `examples/README.md` | Example start-here guide, build commands, maintained program list, writing-your-own notes. | First-use example index and cookbook handoff surface. |
| `benchmarks/README.md` | Benchmark quick navigation, result interpretation, report targets, workflow groups, maintained category split, CLI details, CSV schemas. | Local-measurement and generated-report interpretation surface. |

## Surface Classification

| Classification | Primary surfaces | Notes |
| --- | --- | --- |
| First-use guide | `README.md`, `docs/tutorial.md`, `examples/README.md` | These should answer what to read, build, and run first without forcing users through maintainer history. |
| Concise reference | `docs/solver_selection.md`, `docs/matrix_market.md`, selected sections of `docs/algorithm.md` | These should describe current behavior, assumptions, API routing, and limitations. |
| Generated-report index | `benchmarks/README.md`, Sprint 131 report-index artifacts, large-matrix guardrail `index.tsv` references | These should explain source, freshness, schema, and interpretation boundaries. |
| Maintainer history | `docs/maintainer_guide.md`, planning artifacts, sprint-era snapshots in `docs/algorithm.md` | These should stay reachable but should not be the default first-use path. |
| Historical measurement appendix | Reorder/fill measurements, sprint closure notes, report-gate history, benchmark-governance history currently embedded in `docs/algorithm.md` | These should move behind an explicit appendix or historical context route. |

## Overlap and Duplication List

| Overlap | Surfaces | Audit result |
| --- | --- | --- |
| Workflow choice appears in several places | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `examples/README.md` | Useful, but the current path can make users bounce between four guides before selecting an example. |
| Build and linking setup repeats | `README.md`, `INSTALL.md`, `docs/tutorial.md`, `examples/README.md` | `INSTALL.md` should remain authoritative; adoption docs should link to it instead of restating support tiers in detail. |
| Example handoff repeats | `README.md`, `docs/solver_selection.md`, `docs/tutorial.md`, `examples/README.md` | Keep `examples/README.md` as the maintained program index and make other docs route there by workflow family. |
| Benchmark interpretation repeats | `README.md`, `docs/solver_selection.md`, `docs/tutorial.md`, `docs/algorithm.md`, `benchmarks/README.md` | `benchmarks/README.md` should remain the authoritative measurement surface; adoption docs should link to it only after API path selection. |
| Support-tier and non-claim wording repeats | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `benchmarks/README.md` | Necessary, but first-use docs should stay compact and preserve `INSTALL.md`/maintainer guide as detailed support authorities. |
| Algorithm history and measurement context mix with reference | `docs/algorithm.md`, planning artifacts, `benchmarks/README.md` | Highest-risk simplification target; historical measurements should move out of the concise algorithm reference. |

## First-Use Friction List

| Friction | Impact | Candidate follow-up |
| --- | --- | --- |
| `docs/algorithm.md` is too broad for a current reference. | Users looking for current solver behavior must pass through sprint-era implementation and measurement history. | Day 3-6 split into current reference and historical measurement appendix. |
| Compressed-first workflows are mentioned across several docs but lack a single cookbook route. | Users with CSR/CSC or Matrix Market input must infer the direct path from README, tutorial, solver selection, Matrix Market docs, and examples. | Day 7-9 cookbook design and implementation. |
| Benchmark guidance appears before users may have selected an API workflow. | New users can confuse measurement surfaces with correctness or adoption surfaces. | Keep benchmark/report docs as post-adoption measurement handoff. |
| Maintainer guide is linked from first-use docs for quality policy context. | Useful for maintainers, but can pull first-use readers into support-history material too early. | Keep maintainer links available but not on the primary cookbook path. |
| Installed-consumer guidance sits near local examples. | Users may conflate local example build-tree usage with installed package consumption. | Keep `examples/cmake_example/` routed from install docs and a concise examples note. |

## Compressed-First Discoverability Gaps

| Workflow family | Current discoverability | Gap |
| --- | --- | --- |
| Direct solve from compressed input | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, and `examples/example_compressed_input.c` mention CSR/CSC or compressed-first construction. | Needs one direct cookbook route from compressed arrays to direct solve and cleanup ownership. |
| Iterative solve | `docs/tutorial.md`, `docs/solver_selection.md`, `examples/example_iterative.c`, `examples/example_ic_minres.c`, and `examples/example_matrix_free.c` cover pieces. | Needs one path that starts from input shape, picks iterative solver/preconditioner, and links diagnostics. |
| Matrix Market | `docs/matrix_market.md`, `docs/solver_selection.md`, `docs/tutorial.md`, and `examples/example_matrix_market.c` cover loading and use. | Needs one route from `.mtx` load to compressed/public matrix use and solver handoff. |
| SVD and low-rank | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/algorithm.md`, and `examples/example_svd_lowrank.c` cover behavior. | Needs a concise SVD cookbook route that avoids embedding dense workspace or historical measurement details. |
| Eigensolver | `README.md`, `docs/solver_selection.md`, `docs/algorithm.md`, `examples/example_eigs.c`, and `benchmarks/README.md` cover behavior and measurement. | Needs a first-use eigensolver route separated from backend benchmark and sprint-history detail. |
| Benchmark and reports | `benchmarks/README.md`, `docs/solver_selection.md`, and Sprint 131 report artifacts cover measurement/report interpretation. | Needs concise adoption language that says when to move from examples to measurement and how to read generated indexes. |

## Maintainer-History Extraction Candidates

| Candidate | Current location | Reason |
| --- | --- | --- |
| Reorder/fill measurement narratives | `docs/algorithm.md` reorder sections | Useful historical evidence, but too detailed for current algorithm reference. |
| Sprint-labeled implementation closures | `docs/algorithm.md` Cholesky, LDLT, AMD/ND, eigensolver sections | Better as historical appendix or planning links after current behavior summaries. |
| Performance regression gate history | `docs/algorithm.md` and `benchmarks/README.md` | Keep measurement semantics in benchmark docs and historical rationale in appendix. |
| Backend and sentinel report interpretation history | `docs/algorithm.md`, `benchmarks/README.md`, Sprint 131 artifacts | Adoption docs should summarize boundaries, not duplicate full report-governance history. |
| Historical support snapshots | `docs/maintainer_guide.md` | Keep in maintainer guide; avoid moving into first-use docs. |

## Link and Path Dependencies

| Dependency | Must preserve |
| --- | --- |
| `README.md` links to `docs/tutorial.md`, `docs/solver_selection.md`, `INSTALL.md`, `examples/README.md`, `benchmarks/README.md`, and `docs/maintainer_guide.md`. | Front-door routing must keep these destinations valid or update them together. |
| `docs/algorithm.md` currently links back to README, solver selection, examples, and benchmark docs near the top. | Any split must keep current-reference and historical-appendix entry points discoverable. |
| `docs/solver_selection.md` links to benchmark docs and example names. | Cookbook work must keep solver-selection handoffs coherent. |
| `docs/tutorial.md` links to install, benchmark, solver-selection, and Matrix Market guidance. | Cookbook changes must avoid breaking the guided tutorial path. |
| `examples/README.md` names maintained example binaries and source files. | Cookbook links should target these maintained examples rather than duplicating code. |
| `benchmarks/README.md` names report directories and generated `manifest.txt`/`index.tsv` files. | Report-index docs must preserve generated-report freshness and schema context. |
| `INSTALL.md` owns package/install support details. | Adoption docs should link to install support rather than duplicating platform-tier tables. |

## Day 3 Handoff

Day 3 should design the algorithm-document split:

- choose the concise current-reference target and historical measurement
  appendix target;
- map `docs/algorithm.md` sections into keep, move, summarize, or link-only
  categories;
- preserve top-of-file adoption routing and backlinks from README, solver
  selection, benchmark docs, and maintainer guide;
- explicitly keep benchmark/report measurements and sprint-era rationale out
  of default first-use guidance.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every major adoption document has an assigned role. | Complete | Audited-surfaces and surface-classification tables assign current and target roles. |
| Duplicated or displaced material is captured before edits begin. | Complete | Overlap, friction, and maintainer-history extraction tables name the primary duplicated areas. |
| Compressed-first discoverability gaps are named by workflow family. | Complete | Gap table covers direct, iterative, Matrix Market, SVD, eigensolver, benchmark, and report workflows. |
