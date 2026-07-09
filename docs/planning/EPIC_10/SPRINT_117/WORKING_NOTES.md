# Sprint 117 Working Notes

## Sprint Goal

Sprint 117 closes Epic 10 by integrating the final evidence set, validating
reviewed surfaces, calibrating public claims against the Sprint 100
state-of-the-art target, publishing residuals and non-claims, and writing the
Sprint 117 and Epic 10 retrospectives.

## Starting Constraints

- Treat Sprint 100 as the claim and evidence contract for Epic 10 closeout.
- Treat Sprints 114, 115, and 116 as guardrails for proof-owner,
  package/platform, ABI, Windows, adoption-surface, and public-claim residuals.
- Do not add implementation, source movement, package/platform support, ABI
  support, package-manager recipes, or new benchmark semantics unless the
  Sprint 117 claim audit explicitly requires it and the matching validation is
  captured.
- Do not promote broad state-of-the-art, SuiteSparse/PETSc/Trilinos parity,
  portable performance superiority, shared-library ABI, package-manager, or
  symmetric platform-parity claims without new evidence.
- If documentation only changes, run `git diff --check` and a focused
  trailing-whitespace scan over touched documentation. If code, workflow,
  Make/CMake, script, or package surfaces change, run the relevant validation
  lane before proceeding. If `.c` or `.h` files change, run
  `make format && make lint && make test`.

## Closeout Guardrails From Prior Sprints

| Source | Guardrail | Sprint 117 handling |
|---|---|---|
| Sprint 100 Day 6 target | Epic 10 targets product-grade, self-contained C sparse linear algebra maturity with compressed-first workflows, external evidence, support tiers, and calibrated claims. | Use as the final comparison target, not as permission for broad replacement claims. |
| Sprint 100 Day 13 register | Candidate and blocked claims require implementation or documentation changes, filled evidence, validation, skipped-case records, and remaining non-claims. | Require an evidence source for every earned claim and downgrade unsupported claims. |
| Sprint 114 retrospective | Eigensolver source movement, direct/iterative oracle sharing, and broad SVD helper abstraction remain deferred without exact source-list, CMake, consumer, CTest, and rollback proof. | Keep proof-owner/source-boundary work residual unless Sprint 117 only documents the residual. |
| Sprint 115 retrospective | Linux install CI, macOS install/export parity, Windows install validation, Windows thread/fuzz/property parity, shared-library ABI, and package-manager support remain unclaimed. | Preserve package/platform support tiers and non-claims during final public claim audit. |
| Sprint 116 retrospective | Adoption surfaces were cleaned for claim boundaries; algorithm split and generated benchmark indexes remain possible future work. | Use adoption QA artifacts as public wording guardrails and residual queue inputs. |

## Evidence Surface Inventory

| Surface | Primary sources | Sprint 117 use |
|---|---|---|
| Implementation and APIs | `include/`, `src/`, `tests/`, Sprint 101-114 artifacts | Check whether final implementation evidence earns or limits product-maturity claims. |
| Solver and external comparisons | Sprint 102-103 artifacts, comparison scripts, solver tests, external-oracle notes | Build the final comparison package and identify bounded solver-family evidence. |
| Benchmarks and performance | `benchmarks/`, benchmark reports, Sprint 104-105 artifacts, Sprint 116 benchmark QA | Preserve local-calibration wording and avoid portable performance claims. |
| Package and platform support | `INSTALL.md`, `CMakeLists.txt`, install tests, workflow docs, Sprint 112 and Sprint 115 artifacts | Confirm support tiers, expected exclusions, static-first package truth, and ABI non-claims. |
| Maintainability and source boundaries | source-list artifacts, Sprint 106-110 and Sprint 114 notes | Capture large-owner and proof-owner progress without claiming unresolved source movement. |
| Documentation and adoption | `README.md`, `INSTALL.md`, `docs/`, `examples/`, `benchmarks/README.md`, Sprint 111 and Sprint 116 artifacts | Audit public wording and remove or fence unsupported claims. |
| Residual queue | Sprint 114-116 retrospectives, Sprint 117 artifacts | Publish the post-Epic residual queue and explicit non-claims. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Final integration intake, evidence map, validation-lane inventory, and owner map. | Item 1 |
| 2 | End-state claim inventory against the Sprint 100 target and evidence contract. | Item 1 |
| 3 | Earned-claim, downgrade, and non-claim decisions before cleanup. | Item 1 |
| 4 | Full validation lane design for reviewed and supplemental closeout checks. | Item 2 |
| 5 | Full validation execution and failure triage. | Item 2 |
| 6 | Final validation package and evidence summary. | Item 2 |
| 7 | Final comparison package inventory for solver, reorder, benchmark, coverage, and package evidence. | Item 3 |
| 8 | Final comparison calibration and unsupported-claim cleanup. | Items 3, 4 |
| 9 | Residual queue intake across Sprint 100-116 deferred debt. | Item 5 |
| 10 | Residual queue publication and final non-claim register. | Item 5 |
| 11 | Sprint 117 retrospective draft. | Item 6 |
| 12 | Sprint 117 retrospective finalization and validation. | Item 6 |
| 13 | Epic 10 retrospective draft and cross-sprint synthesis. | Item 7 |
| 14 | Epic 10 retrospective finalization, post-epic handoff, and closeout validation. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation only | `git diff --check`; focused trailing-whitespace scan over touched documentation; local Markdown link checks when links change. |
| Public claims or support wording | Evidence-source cross-check against Sprint 100 target, Sprint 114-116 guardrails, and current validation artifacts. |
| Code or headers | `make format && make lint && make test`; add focused tests or proof artifacts for changed behavior. |
| Makefile, CMake, install, package, workflow, or script surfaces | Run the affected build, install, package, CMake, or workflow-equivalent local checks and record results. |
| Benchmarks or reports | Regenerate the affected report or explicitly record why existing evidence remains valid; preserve local-timing caveats. |
| Platform-support wording | Check against reviewed lane evidence, expected counts, staged exclusions, and package/platform non-claims. |

## Day 1 Notes

- Created the Sprint 117 working-notes baseline and artifact directory.
- Re-read the Sprint 117 plan and confirmed the closeout sequence maps to all
  seven project-plan items.
- Re-read Sprint 100 state-of-the-art and claim/non-goal artifacts:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day6-state-of-the-art-target.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-claim-non-goal-register.md`
- Inventoried Sprint 100-116 retrospectives as the closeout evidence spine.
- Re-read Sprint 114, Sprint 115, and Sprint 116 residual-debt sections to
  preserve deferred proof-owner, package/platform, ABI, Windows, and adoption
  non-claims.
- Built the Day 1 evidence map, validation-lane inventory, and day-level owner
  map in `artifacts/day1-final-integration-intake.md`.
- Kept Day 1 documentation-only; no C source, header, build, workflow, package,
  or benchmark artifacts were modified.

## Day 2 Notes

- Re-read the Sprint 117 Day 2 plan and kept the day scoped to inventory and
  classification, not cleanup or claim promotion.
- Re-read the Sprint 100 target, claim/non-goal register, and handoff package
  to extract the end-state claims and promotion rules.
- Inventoried current public/adoption surfaces for claim-bearing wording:
  `README.md`, `INSTALL.md`, `docs/*.md`, `benchmarks/README.md`, and
  `examples/README.md`.
- Reviewed Sprint 101-116 retrospective and artifact evidence for
  compressed-first workflows, solver comparison/oracle evidence,
  backend/runtime behavior, reorder/fill reporting, maintainability,
  package/platform support, adoption QA, residual proof-owner decisions, and
  non-claims.
- Classified each Sprint 100 target claim as earned pending final validation,
  partially earned, deferred, or non-claim candidate for Day 3 decision.
- Identified claims that require final validation before they can remain public
  in Sprint 117:
  - compressed-first product model;
  - bounded direct-solver external oracle evidence;
  - bounded iterative/eigensolver/SVD comparison evidence;
  - backend/runtime observability and performance-sentinel wording;
  - reorder/fill and benchmark reporting claims;
  - source/test maintainability progress;
  - package/platform support tiers;
  - adoption-surface non-claims and public wording guardrails.
- Added Day 2 claim inventory artifact:
  `artifacts/day2-end-state-claim-inventory.md`.

## Day 3 Notes

- Re-read the Day 2 claim inventory and Sprint 100 dependency-aware claim
  model.
- Re-read the Sprint 116 adoption non-claims checklist to align Day 3 public
  claim decisions with the latest adoption-facing guardrails.
- Marked bounded compressed-first workflows, selected direct-solver external
  oracle lanes, API usability/docs, Matrix Market load/save behavior,
  static-first package support, and tiered platform wording as earned pending
  final Sprint 117 validation.
- Marked product-grade maturity, mutable-shell positioning,
  iterative/eigensolver/SVD comparison architecture, backend/runtime
  observability, local performance sentinels, reorder/fill reporting, and
  maintainability/source ownership as bounded/partially earned claims that
  require careful wording and final validation evidence.
- Kept broad replacement, ecosystem parity, portable performance superiority,
  universal solver-family validation, shared-library/dynamic ABI,
  package-manager support, Windows install/Makefile parity, symmetric platform
  parity, broad complex/mixed precision, GPU, and distributed-memory maturity
  as explicit non-claims.
- Cross-checked the decision against README, install, benchmark,
  solver-selection, Matrix Market, algorithm, example, and maintainer-guide
  surfaces from the Day 2 scan and Sprint 116 guardrail artifacts.
- Added Day 3 end-state claim decision artifact:
  `artifacts/day3-end-state-claim-decision.md`.

## Day 4 Notes

- Re-read the Sprint 117 Day 4 plan and the Day 3 claim decision artifact.
- Reviewed Makefile validation targets for reviewed local quality, reviewed
  CMake parity, source-list checks, install scripts, benchmarks, performance
  sentinels, large-matrix guardrails, sanitizer paths, wall-check, and
  coverage.
- Reviewed Linux, macOS, and Windows workflow comments to preserve the
  reviewed/supplemental/staged lane split:
  - Linux is the strongest reviewed source of truth for Makefile
    compile-quality, CMake parity, and dead-code completeness.
  - macOS has an Apple Clang reviewed path plus supplemental Homebrew GCC and
    static-first install confidence.
  - Windows remains the reviewed MSVC CMake-first consumer subset with
    expected CTest count `51` and staged exclusions.
- Selected `make quality-review-full` as the strongest local reviewed baseline
  for Day 5 if runtime/tooling is available.
- Selected conditional supplemental commands for package/install,
  benchmark/report, coverage, sanitizer, source-list, and documentation
  surfaces based on what Sprint 117 touches.
- Added Day 4 full validation design artifact:
  `artifacts/day4-full-validation-design.md`.

## Day 5 Notes

- Re-read the Day 4 validation design and executed the Day 5 checklist.
- Recorded the starting surface:
  - branch: `sprint-117`
  - base commit: `542bd228`
  - changed files: Sprint 117 planning documentation only
  - changed `.c` / `.h` files: `0`
  - changed Make/CMake/workflow/package/script/benchmark/source/test files:
    `0`
- Ran documentation hygiene:
  - `git diff --check`: passed
  - `rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_117`: passed with no
    matches
- Ran the strongest local reviewed baseline:
  - `make quality-review-full`: passed
  - Makefile reviewed path passed:
    `format-check`, `lint`, `test`, and `deadcode-check`
  - CMake reviewed parity path passed:
    configure, clean build, `ctest -N`, Makefile/CMake test-count parity, and
    full CTest
  - CMake registered tests: `54`
  - Makefile/CMake test-count parity: `54` vs `54`
  - CTest execution: `54 / 54` passed, `0` failed, total real time
    `242.37 sec`
- Did not run supplemental install, benchmark, sanitizer, coverage, or
  package lanes because no package/install, benchmark/report, runtime,
  coverage, workflow, build metadata, source, or header surfaces changed.
- Added Day 5 validation execution artifact:
  `artifacts/day5-validation-execution.md`.

## Day 6 Notes

- Re-read the Sprint 117 Day 6 plan and Day 5 validation execution artifact.
- Packaged the validation evidence into a retrospective-ready closeout
  artifact.
- Confirmed the changed-surface matrix:
  - Sprint 117 planning docs changed;
  - `.c`, `.h`, source, include, test, benchmark, Makefile, CMake, workflow,
    package, install, and script surfaces unchanged.
- Confirmed required validation is complete for touched files:
  - docs hygiene passed;
  - strongest local reviewed baseline passed;
  - no C/header-specific quality requirement remains unrun.
- Recorded validation residual risks:
  - platform workflow proof remains CI-owned;
  - supplemental package, benchmark, sanitizer, and coverage lanes were
    intentionally skipped and must not be cited as fresh passing proof;
  - dead-code output remains completeness/supporting context, not a
    zero-finding claim.
- Closed Sprint 117 project-plan Item 2 with final validation package:
  `artifacts/day6-final-validation-package.md`.

## Day 7 Notes

- Re-read the Sprint 117 Day 7 plan, Day 6 validation package, and final
  handoff artifacts from Sprints 102, 103, 104, 105, 112, 113, 114, 115, and
  116.
- Inventoried final comparison surfaces for direct solvers, iterative methods,
  eigensolvers, SVD, backend/runtime sentinels, reorder/fill guardrails,
  package/platform proof, coverage/dead-code evidence, adoption guardrails, and
  final validation.
- Classified each surface as public claim evidence, local measurement context,
  supplemental proof, or residual background before Day 8 public wording
  cleanup.
- Recorded command provenance for the current Sprint 117 validation baseline
  and historic solver, benchmark, guardrail, and package proof lanes.
- Did not regenerate benchmark, coverage, package, or guardrail artifacts
  because Sprint 117 still has no changed implementation, benchmark, package,
  install, workflow, build, script, source, test, or header surfaces.
- Confirmed that current Day 7 evidence does not treat local benchmark or
  performance artifacts as portable performance proof.
- Added Day 7 final comparison inventory artifact:
  `artifacts/day7-final-comparison-inventory.md`.

## Day 8 Notes

- Re-read the Day 3 claim decision, Day 7 comparison inventory, Day 6
  validation package, and Sprint 116 adoption handoff before starting cleanup.
- Rechecked active public/support surfaces for unsupported or over-broad
  wording:
  - `README.md`
  - `INSTALL.md`
  - `docs/*.md`
  - `benchmarks/README.md`
  - `examples/README.md`
- Verified current public/support wording already fences:
  - local benchmark and performance interpretation;
  - static-first package support and tiered platform scope;
  - Windows reviewed CMake subset and no separate install-validation claim;
  - no shared-library, dynamic ABI, or package-manager claim;
  - Matrix Market load/save functions only, not a public Matrix I/O module or
    builder API;
  - bounded solver comparison evidence without ecosystem parity.
- Did not edit public/support docs because the focused cleanup pass found no
  required unsupported-claim correction.
- Packaged the final comparison and no-edit cleanup record in
  `artifacts/day8-final-comparison-cleanup.md`.
- Closed Sprint 117 project-plan Items 3 and 4 for the current
  documentation-only branch state.

## Day 9 Notes

- Re-read the Sprint 117 Day 9 plan and confirmed the day is residual intake,
  not final residual publication.
- Re-read Sprint 114, Sprint 115, and Sprint 116 residual deferred-debt
  sections and their consciously-closed lists.
- Classified Sprint 114 proof-owner, eigensolver source-boundary,
  direct/iterative oracle, SVD helper, and touched-surface validation residuals
  as post-Epic residuals, future-epic candidates, explicit non-claim
  guardrails, or consciously closed work.
- Classified Sprint 115 Linux install CI, macOS install/export, Windows
  install/thread/fuzz/property, shared-library/dynamic ABI, package-manager,
  and Sprint 114 carry-forward residuals as future-epic candidates,
  post-Epic residuals, or consciously closed decision work.
- Classified Sprint 116 algorithm-doc split and generated benchmark index
  residuals as optional scanability work, while preserving its package,
  platform, ABI, Matrix Market, proof-owner, and implementation non-claims.
- Confirmed no residual was promoted during Day 9 and no implementation,
  package/platform, ABI, install, benchmark, workflow, public API, source-list,
  helper-target, or CTest claim changed.
- Added Day 9 residual queue intake artifact:
  `artifacts/day9-residual-queue-intake.md`.

## Day 10 Notes

- Re-read the Sprint 117 Day 10 plan, Day 8 claim-cleanup artifact, and Day 9
  residual intake artifact.
- Published the post-Epic residual queue for eigensolver source-boundary and
  proof-owner carry-forward work.
- Published future-epic candidates for direct/iterative oracle sharing, SVD
  proof-helper ownership, Linux install CI, macOS install/export parity,
  Windows install/thread/fuzz/property work, shared-library/dynamic ABI
  support, and package-manager support.
- Published optional scanability work for the algorithm-document split and
  generated benchmark artifact indexes.
- Published the explicit non-claim register for deferred package/platform,
  proof-owner, source-boundary, oracle, SVD helper, ABI, package-manager,
  platform parity, Matrix Market public-module, benchmark, and performance
  claims.
- Marked Sprint 114-116 completed intake, proof, decision, adoption QA, and
  validation work as consciously closed so it is not duplicated in the
  unresolved queue.
- Closed Sprint 117 project-plan Item 5 with
  `artifacts/day10-residual-queue-and-nonclaims.md`.

## Day 11 Notes

- Re-read the Sprint 117 Day 11 plan, all Sprint 117 artifacts through Day 10,
  and the current working notes.
- Re-read the Sprint 116 retrospective format to keep the Sprint 117 draft
  aligned with prior Epic 10 sprint retrospectives.
- Built a draft definition-of-done checklist covering intake, claim inventory,
  validation, comparison packaging, cleanup, residual intake, and non-claim
  publication.
- Drafted retrospective sections for what went well, what did not go well,
  validation metrics, claim/comparison outcomes, residual deferred debt, key
  deliverables, and consciously closed work.
- Recorded pre-Day-11 draft metrics:
  - `10` artifact files under `SPRINT_117/artifacts/`;
  - `1143` artifact lines before the Day 11 draft;
  - `319` working-notes lines before the Day 11 draft;
  - `434` plan lines.
- Identified Day 12 finalization gaps:
  - create final `RETROSPECTIVE.md`;
  - update metrics after finalization;
  - rerun focused documentation hygiene;
  - prepare Epic 10 retrospective source inventory for Day 13.
- Added Day 11 retrospective draft artifact:
  `artifacts/day11-sprint-retrospective-draft.md`.

## Day 12 Notes

- Re-read the Sprint 117 Day 12 plan and Day 11 retrospective draft artifact.
- Finalized `RETROSPECTIVE.md` from the Day 11 draft.
- Updated final retrospective wording with Day 11 and Day 12 documentation
  hygiene expectations.
- Recorded Sprint 117 closeout metrics for artifacts, working notes, plan
  lines, retrospective files, changed surfaces, and validation state.
- Prepared the Epic 10 retrospective source inventory for Day 13.
- Confirmed Sprint 117 remains documentation-only through Day 12:
  - changed `.c` files: `0`;
  - changed `.h` files: `0`;
  - changed Make/CMake/workflow/package/script files: `0`;
  - changed benchmark/source/test/include files: `0`.
- Closed Sprint 117 project-plan Item 6 with final retrospective and
  `artifacts/day12-sprint-retrospective-finalization.md`.

## Day 13 Notes

- Re-read the Sprint 117 Day 13 plan and the Day 12 Epic retrospective source
  inventory.
- Reviewed the Epic 10 project-plan spine, review/todo origin, Sprint 100-117
  retrospectives, and Sprint 117 final validation, comparison, cleanup, and
  residual artifacts.
- Drafted the Epic 10 retrospective header, summary table, cumulative metrics,
  earned-claim table, unearned/non-claim table, validation evidence, lessons
  learned, and post-epic carry-forward queue.
- Matched the post-epic carry-forward queue to the Day 10 residual publication:
  - `6` post-Epic residuals;
  - `8` future-epic candidates;
  - `2` optional scanability items.
- Recorded draft Epic metrics:
  - `18 / 18` planned sprints complete through Sprint 117 Day 13;
  - `2,906` nominal project-plan hours;
  - `270` sprint artifact files before the Day 13 draft;
  - `4163` sprint retrospective lines before the Day 13 draft;
  - `13161` sprint working-notes lines before the Day 13 draft.
- Added the Day 13 Epic 10 retrospective draft artifact:
  `artifacts/day13-epic10-retrospective-draft.md`.

## Day 14 Notes

- Re-read the Sprint 117 Day 14 plan and Day 13 Epic 10 retrospective draft.
- Created the final Epic 10 retrospective:
  `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md`.
- Reconciled final Epic 10 retrospective claims with:
  - Sprint 117 retrospective;
  - Day 6 final validation package;
  - Day 8 comparison and unsupported-claim cleanup artifact;
  - Day 10 residual queue and non-claims artifact.
- Wrote the final closeout handoff artifact:
  `artifacts/day14-epic10-closeout-handoff.md`.
- Confirmed Sprint 117 remains documentation-only through Day 14:
  - changed `.c` files: `0`;
  - changed `.h` files: `0`;
  - changed Make/CMake/workflow/package/script files: `0`;
  - changed benchmark/source/test/include files: `0`.
- Closed Sprint 117 project-plan Item 7 pending final documentation hygiene
  checks.
