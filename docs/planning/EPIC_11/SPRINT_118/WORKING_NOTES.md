# Sprint 118 Working Notes

## Sprint Goal

Sprint 118 freezes the post-Epic-10 baseline, converts the final Epic 10
residual queue into Epic 11 owners, and defines the claim/evidence rules for
the next hardening cycle.

## Starting Constraints

- Treat Epic 10 closeout as the current truth source for earned claims,
  non-claims, reviewed validation, and residual debt.
- Treat the Epic 11 review and todo as planning inputs, not as permission to
  promote claims before implementation, proof, validation, and public wording
  cleanup exist.
- Keep Sprint 118 as a baseline, residual-conversion, template, and audit
  sprint. Do not perform Sprint 119-127 implementation work during the intake
  and truth-freeze phase.
- Preserve compressed-first workflows as the product center while keeping the
  mutable orthogonal linked-list shell as supported compatibility.
- Do not claim broad state-of-the-art replacement, ecosystem parity, portable
  performance superiority, shared-library ABI support, package-manager support,
  GPU support, distributed-memory support, or symmetric platform parity without
  new evidence and explicit public wording cleanup.
- If documentation only changes, run `git diff --check` and a focused
  trailing-whitespace scan over touched documentation. If code, workflow,
  Make/CMake, script, package, benchmark, or test surfaces change, run the
  relevant validation lane before proceeding. If `.c` or `.h` files change, run
  `make format && make lint && make test`.

## Input Artifact Inventory

| Input | Sprint 118 use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 118 section | Authoritative project-plan items, estimates, deliverables, and sprint goal. |
| `docs/planning/EPIC_11/SPRINT_118/PLAN.md` | Day-by-day execution plan and completion criteria. |
| `docs/planning/EPIC_11/reviews/review-codex-2026-07-09.md` | Post-Epic-10 review findings, current metrics, product gaps, and state-of-the-art assessment. |
| `docs/planning/EPIC_11/reviews/todo-codex-2026-07-09.md` | Step-by-step Epic 11 gap-closure sequence and guiding rules. |
| `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md` | Final Epic 10 validation anchor, earned claims, non-claims, and carry-forward queue. |
| `docs/planning/EPIC_10/SPRINT_117/RETROSPECTIVE.md` | Final integration sprint outcome and immediate closeout residuals. |
| `docs/planning/EPIC_10/SPRINT_117/artifacts/` | Final validation, comparison, residual, and non-claim evidence spine. |
| Prior Epic retrospectives | Deferred work history and duplicate-work guardrails. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Sprint intake, artifact skeleton, input inventory, scope boundaries, and day-level owner map. | Items 1-7 intake |
| 2 | Reviewed and supplemental validation-surface inventory. | Item 1 |
| 3 | Baseline quality recheck execution and evidence capture. | Item 1 |
| 4 | CI-tier, platform, install, package, and support-boundary truth freeze. | Items 1, 3 |
| 5 | Epic 10 residual queue intake and duplicate fence. | Item 2 |
| 6 | Residual owner, dependency, and proof-gate map for Epic 11. | Item 2 |
| 7 | Product truth map design for compressed-first, mutable-shell, solver, package, benchmark, and non-claim categories. | Item 3 |
| 8 | Product truth map completion and evidence cross-reference. | Item 3 |
| 9 | Source/test hotspot metric collection. | Item 4 |
| 10 | Hotspot interpretation and Sprint 119-123 owner handoff. | Item 4 |
| 11 | Evidence template refresh design. | Item 5 |
| 12 | Evidence template refresh implementation and usage notes. | Item 5 |
| 13 | Public claim drift audit against current truth and Epic 11 candidate claims. | Item 6 |
| 14 | Sprint 118 closeout, artifact index, residual deferred debt, and Sprint 119-127 handoff package. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation-only planning artifacts | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_118`. |
| Public claim or support wording | Evidence-source cross-check against Epic 10 closeout, Sprint 117 artifacts, Day 8 product truth map, and Day 13 claim audit. |
| Code or public headers | `make format && make lint && make test`; add focused tests or evidence artifacts for changed behavior. |
| Makefile, CMake, workflow, install, package, script, or benchmark surfaces | Run the relevant focused validation lane and record reviewed versus supplemental status. |
| Benchmark or performance reports | Regenerate or cite the affected report and preserve local-measurement caveats. |
| Platform support wording | Check reviewed CI tier, staged exclusions, expected CTest counts, and package/install proof before changing support claims. |

## Scope Boundaries

Sprint 118 may document, classify, audit, and prepare evidence templates for
future work. It should not implement the following work unless a later day
explicitly identifies it as a documentation-only cleanup required by Sprint 118
truth freezing:

- eigensolver source movement or private-owner extraction;
- direct/iterative oracle implementation;
- SVD, QR, or rank-deficient oracle expansion;
- giant-test splits or source-file extraction;
- package, shared-library, ABI, package-manager, or platform support changes;
- benchmark threshold changes or portable-performance claims;
- public API expansion;
- broad adoption-surface rewrites reserved for later Epic 11 sprints.

## Day 1 Notes

- Created the Sprint 118 working-notes baseline and artifact directory.
- Re-read the Sprint 118 project-plan section and the day-by-day Sprint 118
  plan.
- Re-read the Epic 11 review and gap-closure todo to identify the baseline,
  residual-conversion, state-of-the-art, and non-claim guardrails.
- Re-read the Epic 10 retrospective and Sprint 117 closeout inputs to preserve
  the final validation anchor, earned claims, non-claims, and residual queue.
- Created the Day 1 sprint intake artifact:
  `artifacts/day1-sprint-intake.md`.
- Kept Day 1 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 2 Notes

- Re-read the Sprint 118 Day 2 plan and kept the work scoped to validation
  inventory, not execution.
- Inspected Makefile validation surfaces:
  - `make quality-review-compile`;
  - `make quality-review`;
  - `make quality-review-cmake-compile`;
  - `make quality-review-cmake`;
  - `make quality-review-full`;
  - `make source-list-check`;
  - `make deadcode-check`;
  - package/install, benchmark, sanitizer, OpenMP, TSan, wall-check, and
    coverage targets.
- Inspected CMake install/export and package surfaces in `CMakeLists.txt`,
  `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, `tests/test_install.sh`, and
  `tests/test_cmake_install.sh`.
- Inspected Linux, macOS, and Windows workflow comments to preserve the
  reviewed/supplemental/staged contract:
  - Linux remains the strongest reviewed source of truth for Makefile
    compile-quality, CMake parity, and dead-code completeness.
  - macOS enforces the Apple Clang reviewed path and carries supplemental
    Homebrew GCC plus static-first Make install/`pkg-config` confidence.
  - Windows remains the reviewed MSVC CMake consumer subset with expected
    CTest count `51` and staged exclusions.
- Selected `make quality-review-full` as the Day 3 strongest local reviewed
  baseline when local tooling and runtime permit.
- Selected documentation hygiene as required for the current touched surface:
  `git diff --check` and a focused trailing-whitespace scan over
  `docs/planning/EPIC_11/SPRINT_118`.
- Added Day 2 validation inventory artifact:
  `artifacts/day2-validation-inventory.md`.
- Kept Day 2 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 3 Notes

- Re-read the Sprint 118 Day 3 plan and Day 2 validation inventory.
- Recorded the starting surface:
  - branch: `sprint-118`
  - base commit: `0605d68e`
  - changed files: Sprint 118 planning documentation only
  - changed `.c` files: `0`
  - changed `.h` files: `0`
  - changed Make/CMake/workflow/package/script/test/benchmark files: `0`
- Ran documentation hygiene:
  - `git diff --check`: passed
  - `rg -n '[ \t]+$' docs/planning/EPIC_11/SPRINT_118`: passed with no
    matches
- Ran the strongest local reviewed baseline:
  - `make quality-review-full`: passed
  - Makefile reviewed path passed:
    `format-check`, `lint`, full `test`, and `deadcode-check`
  - CMake reviewed parity path passed:
    configure, clean build, `ctest -N`, Makefile/CMake test-count parity, and
    full CTest
  - CMake registered tests: `54`
  - Makefile/CMake test-count parity: `54` vs `54`
  - CTest execution: `54 / 54` passed, `0` failed, total real time
    `208.17 sec`
- Did not run supplemental install, benchmark, sanitizer, coverage, package, or
  platform workflow lanes because Sprint 118 Day 3 did not modify those
  surfaces.
- Added Day 3 validation execution artifact:
  `artifacts/day3-baseline-quality-recheck.md`.

## Day 4 Notes

- Re-read the Sprint 118 Day 4 plan and the Day 2-3 validation artifacts.
- Reviewed current Linux, macOS, and Windows workflow definitions:
  - `.github/workflows/ci.yml`;
  - `.github/workflows/macos-ci.yml`;
  - `.github/workflows/windows-ci.yml`.
- Re-read public support wording in `README.md` and `INSTALL.md`.
- Confirmed current CI/platform truth:
  - Linux is the strongest reviewed source of truth for Makefile
    compile-quality, CMake parity, and dead-code completeness.
  - macOS enforces Apple Clang reviewed Makefile/CMake paths plus wall-check
    and sanitizer; Homebrew GCC and static-first Make install/`pkg-config`
    remain supplemental.
  - Windows enforces the MSVC CMake consumer subset only with expected CTest
    count `51`; Makefile parity, install-validation parity, and
    thread/fuzz/property lanes remain staged or unclaimed.
- Reconciled package/install wording with current validation evidence:
  - maintained install story remains static-first;
  - `pkg-config` and `find_package(Sparse)` describe the static archive
    surface;
  - shared-library/dynamic ABI and package-manager support remain deferred.
- Identified Sprint 124-125 handoff candidates for package/ABI and platform
  follow-through.
- Added Day 4 CI-tier and platform truth artifact:
  `artifacts/day4-ci-tier-platform-truth.md`.
- Kept Day 4 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 5 Notes

- Re-read the Sprint 118 Day 5 plan and kept the day scoped to residual intake
  and deduplication, not implementation.
- Extracted residuals from:
  - `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md`;
  - `docs/planning/EPIC_10/SPRINT_117/RETROSPECTIVE.md`;
  - `docs/planning/EPIC_11/reviews/review-codex-2026-07-09.md`;
  - `docs/planning/EPIC_11/reviews/todo-codex-2026-07-09.md`;
  - `docs/planning/EPIC_11/PROJECT_PLAN.md`.
- Deduplicated post-Epic residuals, future-epic candidates, optional
  scanability work, and explicit non-claims into Epic 11 owner candidates.
- Classified each residual by category:
  source owner, proof owner, oracle, performance, package/platform, adoption,
  reportability, or claim boundary.
- Mapped already-scheduled work to Sprints 119-127 and marked no immediate
  unscheduled residuals from the Day 5 intake.
- Added Day 5 residual intake artifact:
  `artifacts/day5-residual-intake.md`.
- Kept Day 5 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 6 Notes

- Re-read the Sprint 118 Day 6 plan and Day 5 residual intake artifact.
- Assigned every deduplicated residual candidate to a Sprint 119-127 owner or
  an explicit future-epic deferral bucket.
- Built the dependency order:
  source-boundary proof before oracle/test splits; oracle and corpus decisions
  before report/performance interpretation; package/ABI decisions before
  platform install parity; platform/package truth before adoption wording; all
  owner-sprint outcomes before Sprint 127 claim recalibration.
- Defined proof gates for source movement, oracle sharing, giant-test splits,
  corpus/report indexes, performance sentinels, package/ABI decisions,
  platform support changes, adoption changes, and public-claim promotion.
- Marked package-manager support, Windows Makefile parity, GPU,
  distributed-memory, broad ecosystem parity, broad complex/mixed precision,
  and portable performance superiority as future-epic or explicit non-claim
  candidates unless an owner sprint earns new evidence.
- Added Day 6 residual owner map:
  `artifacts/day6-residual-owner-map.md`.
- Kept Day 6 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 7 Notes

- Re-read the Sprint 118 Day 7 plan and Day 4-6 artifacts.
- Scanned public/adoption evidence sources for product-truth categories:
  `README.md`, `INSTALL.md`, `docs/solver_selection.md`,
  `docs/tutorial.md`, `docs/matrix_market.md`, `benchmarks/README.md`,
  `examples/README.md`, and `docs/maintainer_guide.md`.
- Defined Day 8 truth-map categories:
  compressed-first storage, mutable-shell compatibility, direct solvers,
  iterative solvers, eigensolvers, SVD/QR/rank surfaces, Matrix Market I/O,
  graph/reorder, package/platform, benchmark/performance, validation/reporting,
  adoption/docs, and explicit non-claims.
- Defined classification rules for baseline truth, Epic 11 candidate claims,
  explicit non-claims, and future-epic deferrals.
- Drafted the Day 8 product truth map template and evidence-source inventory.
- Added Day 7 product truth map design artifact:
  `artifacts/day7-product-truth-map-design.md`.
- Kept Day 7 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 8 Notes

- Re-read the Day 7 product truth map design and used its categories,
  classification rules, evidence inventory, and candidate-claim fences as the
  Day 8 schema.
- Filled the product truth map across compressed-first storage, mutable-shell
  compatibility, direct solvers, iterative solvers, eigensolvers, SVD/QR/rank,
  Matrix Market I/O, graph/reorder, package/platform, benchmarks,
  validation/reporting, adoption/docs, and explicit non-claims.
- Cross-checked current baseline statements against Day 3 reviewed local
  validation evidence and Day 4 CI/package/platform truth:
  - `make quality-review-full` passed;
  - CMake registrations and Makefile test count matched at `54` vs `54`;
  - full CTest passed `54 / 54`;
  - Linux remains the strongest reviewed validation source;
  - macOS and Windows remain tiered, with Windows still scoped to the reviewed
    MSVC CMake consumer subset.
- Kept future statements fenced as Epic 11 candidate claims and assigned them
  to Sprints 119-127 rather than promoting them to current product truth.
- Preserved explicit non-claims for broad state-of-the-art replacement,
  ecosystem parity, portable performance superiority, shared-library ABI,
  package-manager support, GPU, distributed memory, symmetric platform parity,
  and broad precision maturity.
- Added Day 8 product truth map artifact:
  `artifacts/day8-product-truth-map.md`.
- Kept Day 8 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 9 Notes

- Re-read the Sprint 118 Day 9 plan and kept the day scoped to metric
  collection, not source movement or test splitting.
- Collected repository file-count metrics across `src`, `include`, `tests`,
  `benchmarks`, `examples`, and `docs`:
  - total files across those surfaces: `2435`;
  - `src`: `68`;
  - `include`: `19`;
  - `tests`: `89`;
  - `benchmarks`: `19`;
  - `examples`: `18`;
  - `docs`: `2222`.
- Collected file-type counts across those surfaces:
  - C source files: `134`;
  - headers: `49`;
  - Markdown files: `1693`;
  - shell scripts: `2`;
  - CMake files: `1`.
- Collected largest source-owner metrics. The largest source owners are:
  `src/sparse_ldlt_csc.c` at `2095` lines,
  `src/sparse_lu_csr.c` at `1594`,
  `src/sparse_ldlt.c` at `1535`,
  `src/sparse_iterative.c` at `1495`,
  `src/sparse_qr.c` at `1448`,
  `src/sparse_eigs.c` at `1412`,
  and `src/sparse_svd.c` at `1319`.
- Collected largest test-owner metrics. The largest proof owners are:
  `tests/test_ldlt_csc.c` at `3915` lines,
  `tests/test_integration.c` at `3279`,
  `tests/test_qr.c` at `3234`,
  `tests/test_ldlt.c` at `3006`,
  `tests/test_etree.c` at `2962`,
  `tests/test_iterative.c` at `2924`,
  and `tests/test_svd.c` at `2823`.
- Captured approximate function/test proxy counts with `rg`-based patterns to
  rank proof-owner density without treating the counts as semantic API totals.
- Identified mixed-responsibility source candidates and giant-test
  proof-owner candidates for Day 10 interpretation and Sprint 119-123 handoff.
- Added Day 9 hotspot metric collection artifact:
  `artifacts/day9-hotspot-metrics.md`.
- Kept Day 9 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 10 Notes

- Re-read the Sprint 118 Day 10 plan, Day 9 hotspot metric artifact, Day 6
  residual owner map, the Epic 11 review, the Epic 11 gap-closure todo, and
  the Sprint 119-123 project-plan sections.
- Interpreted the Day 9 metrics against the review finding that giant tests
  and large source owners remain the largest practical maintainability risk.
- Separated high-risk owners from large-but-coherent owners:
  - highest-risk owners include `tests/test_ldlt_csc.c`,
    `src/sparse_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_svd.c`,
    `src/sparse_eigs.c`, `src/sparse_iterative.c`,
    `tests/test_iterative.c`, and `tests/test_integration.c`;
  - large-but-coherent or defer-first owners include `src/sparse_matrix.c`,
    private direct-solver headers, graph/reorder private headers, benchmark
    drivers, and `docs/algorithm.md`.
- Mapped handoff guidance to Sprints 119-123:
  - Sprint 119: eigensolver source-boundary feasibility and focused consumer
    proof;
  - Sprint 120: direct/iterative oracle and giant-test split work;
  - Sprint 121: SVD, QR, rank, and dense/external reference proof work;
  - Sprint 122: corpus, coverage, and report-index architecture;
  - Sprint 123: performance, backend, graph, reorder, and benchmark-report
    governance.
- Defined source-movement prerequisites and giant-test split prerequisites so
  future sprints receive proof gates rather than broad refactor mandates.
- Added no-move/defer guidance for integration tests, foundational matrix
  compatibility code, internal headers, graph/reorder movement, benchmark
  driver splits, and product-doc restructuring.
- Added Day 10 hotspot owner handoff artifact:
  `artifacts/day10-hotspot-owner-handoff.md`.
- Kept Day 10 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 11 Notes

- Re-read the Sprint 118 Day 11 plan and kept the day scoped to evidence
  template design, not implementation or source/test movement.
- Reviewed the existing Sprint 100 reusable template artifacts:
  - solver comparison evidence template;
  - benchmark, coverage, and performance templates;
  - platform and packaging evidence templates.
- Reviewed recent Sprint 114-117 retrospective and working-note patterns for
  residual deferred debt, validation, proof-owner, non-claim, and handoff
  sections.
- Cross-checked Day 11 needs against Sprint 118 Day 6 proof gates, Day 8
  product truth map, Day 10 hotspot owner handoff, and Sprint 119-127 owners.
- Identified template gaps for:
  - source movement and giant-test splits;
  - oracle expansion and corpus trust boundaries;
  - performance sentinels, report indexes, backend/runtime context, and stale
    report handling;
  - package/ABI decision paths and package-manager disposition;
  - adoption cleanup, link/path checks, and claim-boundary scans.
- Designed the Day 12 refreshed template set:
  - `source-movement-evidence-template.md`;
  - `oracle-expansion-evidence-template.md`;
  - `performance-sentinel-evidence-template.md`;
  - `package-abi-decision-template.md`;
  - `adoption-cleanup-evidence-template.md`;
  - `template-usage-notes.md`.
- Defined required shared fields for scope, baseline, proof values, change
  plan, validation, drift, non-claims, and handoff.
- Added Day 11 evidence template design artifact:
  `artifacts/day11-evidence-template-design.md`.
- Kept Day 11 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 12 Notes

- Re-read the Sprint 118 Day 12 plan and the Day 11 evidence template design.
- Created the reusable template directory:
  `docs/planning/EPIC_11/SPRINT_118/templates/`.
- Published refreshed Epic 11 evidence templates:
  - `templates/source-movement-evidence-template.md`;
  - `templates/oracle-expansion-evidence-template.md`;
  - `templates/performance-sentinel-evidence-template.md`;
  - `templates/package-abi-decision-template.md`;
  - `templates/adoption-cleanup-evidence-template.md`;
  - `templates/template-usage-notes.md`.
- Ensured each evidence template includes scope, baseline, proof values,
  validation, drift, non-claims, and residual handoff sections.
- Added usage notes with template selection rules, required Sprint 118 inputs,
  validation rules by touched surface, claim discipline, and Sprint 119-127
  owner mapping.
- Preserved Day 11's boundary that these are blank reusable templates, not
  owner-sprint implementation work.
- Added Day 12 template refresh artifact:
  `artifacts/day12-evidence-template-refresh.md`.
- Kept Day 12 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 13 Notes

- Re-read the Sprint 118 Day 13 plan and the Day 8 product truth map.
- Audited public and support surfaces for public-claim drift:
  - `README.md`;
  - `INSTALL.md`;
  - `docs/solver_selection.md`;
  - `docs/tutorial.md`;
  - `docs/matrix_market.md`;
  - `docs/algorithm.md`;
  - `docs/maintainer_guide.md`;
  - `benchmarks/README.md`;
  - `examples/README.md`.
- Scanned for risky support, performance, parity, platform, package,
  Matrix Market, solver-family, and state-of-the-art wording.
- Compared findings against the Day 8 baseline claim list, candidate claim
  list, explicit non-claim list, Day 4 platform truth, and Day 12 adoption
  cleanup template.
- Found no immediate unsupported public claim requiring a Sprint 118 edit.
- Recorded candidate-only or partially supported areas for future owners:
  - clearer compressed-first product identity and adoption scanability for
    Sprint 126;
  - eigensolver source-boundary and oracle wording after Sprints 119-122;
  - benchmark/performance governance after Sprint 123;
  - package/ABI decisions after Sprint 124;
  - platform validation after Sprint 125;
  - final earned/non-earned claim table after Sprint 127.
- Added Day 13 public claim drift audit artifact:
  `artifacts/day13-public-claim-drift-audit.md`.
- Kept Day 13 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 14 Notes

- Re-read the Sprint 118 Day 14 plan and reviewed all Sprint 118 artifacts and
  templates.
- Confirmed Sprint 118 deliverables now have artifacts:
  - post-Epic-10 baseline package;
  - Epic 11 residual owner map;
  - current product truth map;
  - source/test hotspot metrics;
  - refreshed evidence templates;
  - public claim drift audit;
  - Sprint 119-127 handoff requirements.
- Summarized validation evidence:
  - Day 3 `make quality-review-full` passed;
  - CMake registered tests: `54`;
  - Makefile/CMake parity: `54` vs `54`;
  - CTest passed `54 / 54`;
  - later days remained documentation-only.
- Summarized product truth, explicit non-claims, residual owners, hotspot
  handoffs, refreshed template usage, and claim-drift recommendations.
- Identified residual deferred debt preserved for Sprints 119-127 and
  future-epic/non-claim candidates.
- Added Day 14 closeout and handoff artifact:
  `artifacts/day14-sprint-closeout-handoff.md`.
- Kept Day 14 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.
