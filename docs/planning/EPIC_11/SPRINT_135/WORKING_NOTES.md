# Sprint 135 Working Notes

## Sprint Goal

Simplify the adoption surface after Epic 10 by separating first-use guides
from maintainer history and making compressed-first workflows easier to find.

Sprint 135 follows the Sprint 131 report-index decisions and the Sprint
133-134 package/platform support truth. It must not turn generated report
indexes, benchmark rows, local measurements, static package proof, or
supplemental platform jobs into broader correctness, performance, package,
ABI, or platform claims.

## Starting Constraints

- Treat Sprint 131 report-index policy as the report baseline: generated
  indexes are traceability and freshness evidence, not broad correctness,
  coverage-completeness, or performance proof.
- Treat the existing large-matrix guardrail `index.tsv` as the first accepted
  generated index path; do not imply a normalized cross-report schema exists.
- Treat coverage reports as supplemental and tree-mutating; coverage
  percentage is not reviewed behavioral completeness.
- Treat dead-code reports as conservative report-completeness evidence, not
  zero-findings or removal-ready proof.
- Treat benchmark rows and local measurement output as local evidence, not
  portable performance guarantees.
- Treat Sprint 133 as the static-first package baseline: shared-library
  packaging, dynamic ABI compatibility, runtime-loader behavior, and
  package-manager support remain deferred non-claims.
- Treat Sprint 134 as the platform-tier baseline: Linux owns reviewed
  static-first package-contract CI; macOS package install/export confidence is
  supplemental; Windows install/downstream confidence is supplemental; Windows
  staged pthread/POSIX tests remain staged.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and a focused
  markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_135`.

## Input Artifact Inventory

| Input | Role in Sprint 135 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 135 | Defines the seven Sprint 135 items for adoption audit, algorithm split, cookbook, report docs, validation, and closeout. |
| `docs/planning/EPIC_11/SPRINT_135/PLAN.md` | Provides day-level execution order and 164-hour budget. |
| `docs/planning/EPIC_11/SPRINT_131/artifacts/day14-closeout-report-index-handoff.md` | Provides report-index decisions, freshness rules, generated-versus-curated boundaries, and report non-claims. |
| `docs/planning/EPIC_11/SPRINT_133/artifacts/day14-closeout-package-abi-handoff.md` | Provides static-first package, ABI, shared-library, package-manager, and package-proof boundaries. |
| `docs/planning/EPIC_11/SPRINT_134/artifacts/day14-platform-tier-closeout-handoff.md` | Provides Linux, macOS, Windows, staged-test, and supplemental package/platform support tiers. |
| `README.md` | Front-door adoption path, high-level feature summary, CI/package support summary, and link hub. |
| `INSTALL.md` | Installation, package-consumer, and platform support truth. |
| `docs/tutorial.md` | First-use tutorial and workflow introduction. |
| `docs/solver_selection.md` | Solver-choice guidance and adoption decision support. |
| `docs/algorithm.md` | Current algorithm reference mixed with historical measurement and implementation context. |
| `docs/matrix_market.md` | Matrix Market input and compressed representation guidance. |
| `docs/maintainer_guide.md` | Maintainer history, support-tier ownership, validation ownership, and non-claim boundaries. |
| `examples/README.md` | Example index and adoption entry point for maintained examples. |
| `examples/*.c` | Maintained direct, iterative, Matrix Market, SVD, eigensolver, and compressed-input example sources. |
| `benchmarks/README.md` | Benchmark entry point and local measurement interpretation surface. |
| `benchmarks/*.c` | Maintained benchmark programs and benchmark workflow references. |

## Candidate Adoption Surfaces

| Surface | Current role | Day owner |
| --- | --- | --- |
| `README.md` | Front-door feature, build, package, CI, and documentation index. | Days 2, 10-14 |
| `INSTALL.md` | Static-first install and platform support guidance. | Days 2, 11-14 |
| `docs/tutorial.md` | First-use walkthrough candidate. | Days 2, 8, 11, 13 |
| `docs/solver_selection.md` | Solver adoption decision guide. | Days 2, 6, 8-9, 11, 13 |
| `docs/algorithm.md` | Candidate for concise reference plus historical appendix split. | Days 2-6, 12-14 |
| `docs/matrix_market.md` | Matrix Market and compressed-input adoption surface. | Days 2, 7-9, 11, 13 |
| `docs/maintainer_guide.md` | Maintainer history, validation ownership, and support-tier truth. | Days 2-3, 10-14 |
| `examples/README.md` | Maintained example discovery and cookbook link surface. | Days 2, 7-9, 11, 13 |
| `examples/example_basic_solve.c` | Direct solver first-use source. | Days 7-8, 13 |
| `examples/example_compressed_input.c` | Compressed-input setup source. | Days 7-8, 13 |
| `examples/example_matrix_market.c` | Matrix Market source. | Days 7-8, 13 |
| `examples/example_iterative.c`, `examples/example_ic_minres.c` | Iterative and preconditioned iterative sources. | Days 7-8, 13 |
| `examples/example_svd_lowrank.c` | Low-rank SVD source. | Days 7, 9, 13 |
| `examples/example_eigs.c` | Eigensolver source. | Days 7, 9, 13 |
| `benchmarks/README.md` | Benchmark and local measurement entry point. | Days 2, 9-10, 13 |
| `benchmarks/bench_*.c` | Maintained benchmark workflow references. | Days 7, 9-10, 13 |
| Sprint 131 report artifacts | Report-index and freshness policy source. | Days 10, 12-14 |
| Sprint 133-134 support artifacts | Package, ABI, and platform claim-boundary source. | Days 1, 11-14 |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Sprint intake, artifact baseline, adoption-surface inventory, and claim fences | Items 1-7 |
| 2 | Adoption surface audit | Item 1 |
| 3 | Algorithm split design | Item 2 |
| 4 | Algorithm split preparation and scope decision | Items 2-3 |
| 5 | Algorithm split implementation batch 1 | Item 3 |
| 6 | Algorithm split implementation batch 2 and duplication cleanup | Item 3 |
| 7 | Compressed-first cookbook design | Item 4 |
| 8 | Compressed-first direct, iterative, and Matrix Market cookbook batch | Item 4 |
| 9 | Compressed-first SVD, eigensolver, and benchmark cookbook batch | Item 4 |
| 10 | Benchmark/report index adoption docs | Item 5 |
| 11 | Adoption navigation alignment | Items 1-5 |
| 12 | Link and claim validation | Item 6 |
| 13 | Integrated adoption review | Items 4-6 |
| 14 | Closeout, metrics, residual queue, and handoff | Item 7 |

## Validation Expectations

| Change type | Required validation |
| --- | --- |
| Documentation-only Sprint 135 artifacts | `git diff --check` and focused markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_135`. |
| Adoption docs edits | `git diff --check`, focused trailing-whitespace scan, and link/path checks for touched docs. |
| Documentation movement or split | Link/path scan for old and new targets, inbound-link scan, and claim-boundary scan. |
| Example index or cookbook links | Verify referenced files exist and examples remain maintained sources. |
| Benchmark/report wording | Claim scan for unsupported portable performance, correctness, coverage-completeness, or backend claims. |
| Package/platform wording | Claim scan against Sprint 133-134 static-first and platform support tiers. |
| Script, build, CMake, C, or header edits | Focused syntax/build validation plus `make format && make lint && make test` if any `.c` or `.h` file changes. |

## Scope Boundaries

- Sprint 135 may simplify navigation, move documentation, add cookbook
  guidance, and clarify adoption paths.
- Sprint 135 must not change solver behavior, package behavior, CI support
  tiers, generated report schemas, benchmark semantics, or platform claims
  without explicit implementation and validation.
- Concise adoption language must link to maintained examples or current
  references instead of duplicating source-level implementation detail.
- Historical measurement material should remain reachable, but it should not
  be the default first-use path.
- Report-index guidance must preserve generated-versus-curated and freshness
  semantics from Sprint 131.
- Benchmark and local-measurement guidance must avoid portable performance,
  backend parity, broad scalability, or correctness-over-time claims.
- Package and install wording must preserve Sprint 133 static-first
  non-claims and Sprint 134 platform support tiers.

## Day 1 Notes

- Created the Sprint 135 working-notes baseline and artifact directory.
- Re-read the Sprint 135 project-plan section and mapped Items 1-7 to
  day-level owners.
- Reviewed Sprint 131 closeout as the inherited report-index and freshness
  baseline.
- Reviewed Sprint 133 closeout as the inherited static-first package and ABI
  baseline.
- Reviewed Sprint 134 closeout as the inherited Linux/macOS/Windows platform
  support-tier baseline.
- Inventoried current adoption documents: `README.md`, `INSTALL.md`,
  `docs/tutorial.md`, `docs/solver_selection.md`, `docs/algorithm.md`,
  `docs/matrix_market.md`, `docs/maintainer_guide.md`, `examples/README.md`,
  and `benchmarks/README.md`.
- Inventoried maintained example sources for direct, compressed-input,
  Matrix Market, iterative, SVD, eigensolver, and benchmark adoption paths.
- Recorded validation expectations for documentation-only changes,
  documentation movement, example/cookbook links, benchmark/report wording,
  package/platform wording, and any future C/header edits.
- Preserved inherited claim fences: report indexes are traceability evidence,
  benchmarks are local measurement evidence, package support is static-first,
  and macOS/Windows package install/export confidence remains supplemental.

## Day 2 Notes

- Wrote the adoption surface audit artifact.
- Re-audited the front-door documents: `README.md`, `INSTALL.md`,
  `docs/tutorial.md`, `docs/solver_selection.md`, `docs/algorithm.md`,
  `docs/matrix_market.md`, `docs/maintainer_guide.md`,
  `examples/README.md`, and `benchmarks/README.md`.
- Classified `README.md`, `docs/tutorial.md`, and `examples/README.md` as
  first-use surfaces with overlapping workflow and build/setup orientation.
- Classified `docs/solver_selection.md`, `docs/matrix_market.md`, and the
  current-behavior portions of `docs/algorithm.md` as concise-reference
  surfaces.
- Classified `benchmarks/README.md` and Sprint 131 report-index artifacts as
  generated-report and local-measurement interpretation surfaces.
- Classified `docs/maintainer_guide.md`, planning artifacts, and sprint-era
  measurement narratives inside `docs/algorithm.md` as maintainer/history
  surfaces.
- Identified the highest-risk overlap: `docs/algorithm.md` mixes current
  algorithm reference with historical sprint measurement, report-index,
  performance-gate, and implementation-decision material.
- Identified adoption friction where compressed-first workflows exist but are
  distributed across README, tutorial, solver selection, Matrix Market docs,
  examples, and benchmark docs rather than one cookbook-style path.
- Recorded compressed-first discoverability gaps by direct, iterative, Matrix
  Market, SVD, eigensolver, and benchmark workflow family.
- Recorded link/path dependencies that Day 3-6 algorithm-document movement
  must preserve.
- No product docs, examples, source files, scripts, workflows, or support
  claims were changed on Day 2 beyond Sprint 135 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 3 Notes

- Wrote the algorithm doc split design artifact.
- Re-audited `docs/algorithm.md` section headings, top-of-file adoption
  routing, and inbound links from maintained adoption docs.
- Confirmed the current public inbound link to `docs/algorithm.md` is the
  README documentation index; Sprint 135 planning artifacts also reference it
  as the selected split target.
- Selected `docs/algorithm.md` as the retained concise current-reference
  target so the existing public path stays valid.
- Selected `docs/algorithm_history.md` as the historical measurement appendix
  target for sprint-era measurements, regression-gate rationale, benchmark
  history, and implementation-decision chronology.
- Classified algorithm sections into keep, summarize-and-link, move, and
  preserve-as-link-only buckets.
- Identified historical-heavy blocks in the Cholesky CSC performance section,
  CSC LDLT scaffolding sections, AMD/ND reorder sections, performance
  regression gates, and eigensolver sprint-history paragraphs.
- Defined redirect/orientation requirements: `docs/algorithm.md` should link
  to the historical appendix near the top, and the appendix should link back
  to current reference, solver selection, examples, and benchmark docs.
- Defined Day 4-6 bounded implementation plan: prepare headings and anchors,
  move the highest-risk historical blocks first, then clean duplication and
  run link/path plus claim-boundary validation.
- No product docs, examples, source files, scripts, workflows, or support
  claims were changed on Day 3 beyond Sprint 135 planning artifacts.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 4 Notes

- Wrote the algorithm split preparation artifact.
- Created the historical appendix scaffold at `docs/algorithm_history.md`.
- Added a short `docs/algorithm.md` top-of-file pointer to
  `docs/algorithm_history.md` without moving large blocks yet.
- Preserved `docs/algorithm.md` as the stable current-reference path so the
  README documentation index remains valid.
- Confirmed both target files exist: `docs/algorithm.md` and
  `docs/algorithm_history.md`.
- Re-ran the algorithm heading inventory after adding the appendix scaffold.
- Re-ran the inbound-link scan across README, INSTALL, docs, examples,
  benchmarks, and planning docs; maintained adoption docs have no
  heading-specific inbound links that block the split.
- Selected a bounded first phase rather than a full rewrite: Days 5-6 should
  move the highest-friction historical blocks while leaving broad
  reorganization and cookbook integration for later Sprint 135 days.
- Assigned Day 5 movement candidates to Cholesky/CSC performance history,
  LDLT sprint history, and the AMD/ND reorder chronology.
- Assigned Day 6 movement candidates to benchmark/report gate history and
  eigensolver sprint-history paragraphs, followed by duplication cleanup and
  validation.
- Recorded risks around broken anchors, duplicated claims, stale performance
  phrasing, and historical measurements remaining in the current reference.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 5 Notes

- Wrote the first algorithm split implementation artifact.
- Implemented the bounded Batch 1 split in `docs/algorithm.md` and
  `docs/algorithm_history.md`.
- Kept `docs/algorithm.md` as the stable current-reference path and replaced
  high-friction historical blocks with concise current summaries plus links
  into the appendix.
- Moved/isolate-summarized Cholesky fill comparison history, CSC Cholesky
  performance history, supernodal Cholesky proof trail, CSC LDLT scaffolding,
  supernodal LDLT history, row-adjacency benchmark impact, AMD quotient-graph
  chronology, ND Sprint 22-28 chronology, and retired Pres_Poisson target
  context into `docs/algorithm_history.md`.
- Updated the README documentation index label from Algorithm Description to
  Algorithm Reference while preserving the `docs/algorithm.md` link target.
- Preserved benchmark/report gate history and eigensolver sprint-history
  paragraphs for Day 6 rather than mixing too many movement classes into Day
  5.
- Rechecked that moved historical material remains reachable through appendix
  section links and planning artifact links.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 6 Notes

- Wrote the second algorithm split implementation artifact.
- Completed the bounded algorithm split phase selected on Day 4.
- Replaced the long `Performance regression gates` chronology in
  `docs/algorithm.md` with a concise current note that points to
  `benchmarks/README.md` and the historical appendix.
- Removed sprint labels and rollout chronology from the public symmetric
  eigensolver heading and introduction while preserving current backend/API
  behavior.
- Replaced OpenMP reorthogonalization, convergence heuristic, benchmark
  sweep, shift-invert, and LOBPCG rollout history in `docs/algorithm.md` with
  current summaries and appendix links.
- Expanded `docs/algorithm_history.md` with benchmark/report governance,
  wall-check, performance-sentinel, report-index boundary, eigensolver
  backend rollout, OpenMP reorthogonalization, thick-restart, shift-invert,
  and LOBPCG history.
- Preserved `benchmarks/README.md` as the current benchmark/report command
  and interpretation authority.
- Recorded residual algorithm-doc work for later navigation/cookbook days:
  broad reference reordering, direct cookbook integration, and any remaining
  isolated historical anecdotes outside the Day 5-6 split scope.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 7 Notes

- Wrote the compressed-first cookbook design artifact.
- Audited current compressed-first adoption surfaces across `README.md`,
  `docs/tutorial.md`, `docs/solver_selection.md`, `docs/matrix_market.md`,
  `examples/README.md`, maintained example sources, and
  `benchmarks/README.md`.
- Selected `docs/cookbook.md` as the target first-use cookbook page for Day 8
  and Day 9 implementation.
- Designed cookbook sections for starting from CSR/CSC/Matrix Market data,
  direct solves, iterative solves, Matrix Market load/use, SVD/low-rank,
  symmetric eigensolvers, and benchmark/report handoff.
- Split the implementation queue so Day 8 owns direct, iterative, and Matrix
  Market cookbook content, while Day 9 owns SVD, eigensolver, and benchmark
  cookbook content.
- Kept `benchmarks/README.md` as the benchmark command/report authority and
  kept examples as runnable handoffs rather than copying full source into the
  cookbook.
- Recorded claim boundaries for package-manager availability, shared-library
  ABI, portable performance, state-of-the-art parity, nonsymmetric
  eigensolvers, compressed-array ownership, benchmark pass/fail semantics, and
  repeated-run handle coverage.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 8 Notes

- Implemented the first compressed-first cookbook batch in `docs/cookbook.md`.
- Added concise adoption paths for:
  - starting from caller-owned CSR arrays
  - starting from caller-owned CSC arrays
  - starting from Matrix Market files
  - direct one-shot solves after compressed construction
  - stable-pattern repeated direct reuse after compressed construction
  - iterative solves after compressed construction
  - Matrix Market load/use routing into the same public matrix shell
- Linked the cookbook from `README.md`, `docs/tutorial.md`,
  `docs/solver_selection.md`, and `examples/README.md`.
- Kept maintained examples as runnable handoffs rather than copying complete
  example sources into the cookbook.
- Preserved `docs/matrix_market.md` as the format and errno authority and
  `benchmarks/README.md` as the measurement authority.
- Left SVD, eigensolver, and benchmark/report cookbook expansion for Day 9 as
  planned.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 9 Notes

- Implemented the second compressed-first cookbook batch in `docs/cookbook.md`.
- Added concise adoption paths for:
  - SVD and low-rank workflows after CSR, CSC, or Matrix Market input enters
    the public matrix shell
  - symmetric eigensolver workflows after compressed or loaded input
  - benchmark and generated-report handoff after the API workflow is chosen
- Linked maintained examples and public headers from the cookbook:
  - `examples/example_svd_lowrank.c`
  - `examples/example_eigs.c`
  - `include/sparse_svd.h`
  - `include/sparse_eigs.h`
- Kept `benchmarks/README.md` as the command, CSV, report-artifact, and
  interpretation authority.
- Updated the README documentation-index description for the cookbook so it
  reflects the now-complete workflow coverage.
- Preserved benchmark wording as local measurement guidance rather than a
  portable performance or pass/fail timing claim.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 10 Notes

- Reviewed Sprint 131 report-index requirements, first-index implementation,
  and current benchmark/report docs.
- Added a `Report index handoff` section to `benchmarks/README.md` that names:
  - `build/bench-reports/canonical/index.tsv` and `manifest.txt`
  - `build/bench-reports/sentinels/sentinels.tsv` and `manifest.txt`
  - `build/bench-reports/large-matrix-guardrails/index.tsv` and
    `manifest.txt`
- Added concise report-index interpretation rules for generation command,
  freshness context, row identity, skips, `n/a` fields, fallback context, CSV
  timing scope, and regenerating rather than hand-editing generated rows.
- Updated `docs/cookbook.md` to include the large-matrix guardrail handoff and
  first-use report-index locations.
- Updated `README.md` so the benchmark/report command list surfaces generated
  index/manifest context and the large-matrix guardrail target.
- Preserved Sprint 131 boundaries: report indexes are artifact maps and
  freshness context, not portable performance, scalability, coverage, package,
  platform, or broad pass/fail timing claims.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 11 Notes

- Re-audited top-level navigation after the algorithm split, cookbook
  creation, and report-index documentation pass.
- Added a compact `Adoption Map` to `README.md` that routes first-use readers
  to quick start, solver selection, cookbook, install, benchmark/report,
  algorithm reference, historical appendix, and maintainer policy surfaces.
- Added a `Documentation Map` to `docs/tutorial.md` so the fuller walkthrough
  points to the same owner surfaces without making maintainer history the
  default adoption path.
- Updated `docs/solver_selection.md`, `examples/README.md`,
  `docs/cookbook.md`, `benchmarks/README.md`, and `INSTALL.md` cross-links so
  each page names the layer it owns and hands off to the neighboring owner
  where appropriate.
- Preserved Sprint 133-134 install/package truth by keeping installed consumer
  detail in `INSTALL.md` and avoiding package-manager, shared-library, or
  platform-tier expansion.
- Preserved the Day 5-6 algorithm split by linking current behavior to
  `docs/algorithm.md` and historical measurement notes to
  `docs/algorithm_history.md`.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 12 Notes

- Ran the documentation hygiene and validation sweep across the Sprint 135
  adoption surface.
- Verified `git diff --check` and the trailing-whitespace scan passed.
- Verified local markdown link targets resolve across README, install,
  tutorial, solver-selection, cookbook, algorithm, algorithm history, Matrix
  Market, maintainer, benchmark, and examples docs.
- Verified no `.c` or `.h` files changed.
- Ran package/platform claim scans across README, install docs, cookbook,
  solver-selection, tutorial, algorithm/reference history, maintainer guide,
  benchmark docs, and Sprint 135 artifacts.
- Ran performance/report claim scans for portable-performance, timing-gate,
  report-index, sentinel, canonical-report, and large-matrix guardrail wording.
- Fixed residual public-doc wording found during validation:
  - reframed README LOBPCG preconditioning text as fixture-level local
    evidence rather than a portable speedup guarantee
  - reframed a current-reference RCM fixture speedup phrase as historical
    local measurement context
  - removed remaining sprint-era phrases from eigensolver current-reference
    text in `docs/algorithm.md`
  - moved the LOBPCG LARGEST-preconditioning comparison pointer to the
    historical appendix
- Recorded remaining validation risk as Day 13 integrated walkthrough work,
  not a failed Day 12 check: the adoption path still needs end-to-end reader
  review after all navigation and report-index edits.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 13 Notes

- Walked the README-to-examples-to-cookbook-to-install-to-reference path as a
  first-use reader.
- Walked the benchmark/report-index path from cookbook and benchmark docs
  through canonical, sentinel, and large-matrix guardrail report artifacts.
- Walked the maintainer-history path from README/tutorial to
  `docs/algorithm.md`, `docs/algorithm_history.md`, and
  `docs/maintainer_guide.md`.
- Verified compressed-first direct, iterative, Matrix Market, SVD, eigensolver,
  and benchmark/report paths are all discoverable from `docs/cookbook.md` and
  at least one front-door adoption surface.
- Fixed a naming mismatch by changing `docs/algorithm.md` from "Algorithm
  Description" to "Algorithm Reference" and tightening the intro so current
  reference ownership and historical appendix ownership match README wording.
- Recorded the Day 13 integrated adoption review artifact.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 14 Notes

- Wrote the Sprint 135 closeout and Sprint 136 handoff artifact.
- Reconciled Sprint 135 plan items against artifacts:
  - adoption-surface intake and audit
  - algorithm split design, preparation, and two implementation batches
  - compressed-first cookbook design and two implementation batches
  - benchmark/report-index adoption docs
  - navigation alignment
  - link and claim validation
  - integrated adoption review
- Recorded closeout metrics:
  - 11 public adoption/owner docs checked in validation
  - 2 new public docs: `docs/cookbook.md` and `docs/algorithm_history.md`
  - 6 cookbook workflow families
  - 3 generated report families surfaced
  - 14 Sprint 135 daily artifacts including closeout
  - 0 `.c` / `.h` files changed
- Confirmed final claim boundaries remain aligned with Sprint 133-134 and
  Sprint 131:
  - static-first package/install truth unchanged
  - shared-library, dynamic-ABI, runtime-loader, and package-manager support
    remain non-claims
  - platform support tiers remain unchanged
  - benchmark/report rows remain local measurement or guardrail artifacts
  - generated indexes/manifests remain artifact maps and freshness context
- Prepared the retrospective input queue: docs-only sprint, adoption
  simplification metrics, validation evidence, claim-boundary status, and
  residual follow-up risks.
- No `.c` or `.h` files changed, so the full C quality gate was not required.
