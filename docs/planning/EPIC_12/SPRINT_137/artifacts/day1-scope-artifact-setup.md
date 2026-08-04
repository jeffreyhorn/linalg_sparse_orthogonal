# Sprint 137 Day 1 - Scope & Artifact Setup

## Purpose

Day 1 establishes the Sprint 137 baseline before source/test metrics,
residual reconciliation, gap selection, evidence-template design, quality-map
work, public-claim freeze, or Sprint 138 handoff synthesis begins.

This is a documentation-only setup artifact. It creates the Sprint 137
artifact structure, records inherited inputs, maps project-plan items to
day-level owners, and captures initial validation expectations for later Epic
12 implementation work.

## Sprint 137 Scope

Sprint 137 implements seven project-plan items:

| Item | Project-plan name | Day owner |
| --- | --- | --- |
| 1 | Baseline Metrics Capture | Days 1-3 |
| 2 | Residual Queue Reconciliation | Days 4-5 |
| 3 | Gap Selection Decision | Days 6-7 |
| 4 | Evidence Contract Templates | Days 8-10 |
| 5 | Quality Surface Map | Day 11 |
| 6 | Public Claim Freeze | Day 12 |
| 7 | Sprint Closeout | Days 13-14 |

Day 1 does not choose Epic 12 gaps or implement code. Its completion criteria
are met when the artifact structure, inherited inputs, owner map, and
validation expectations are explicit enough for Day 2 to begin baseline
collection without guessing support tiers.

## Artifact Structure

| Path | Role |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_137/PLAN.md` | Day-by-day Sprint 137 execution plan. |
| `docs/planning/EPIC_12/SPRINT_137/WORKING_NOTES.md` | Sprint goal, constraints, input inventory, day ownership, validation expectations, claim fences, and day notes. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day1-scope-artifact-setup.md` | This Day 1 scope and setup artifact. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day2-source-test-maintainability-baseline.md` | Planned Day 2 source, test, benchmark, example, and maintainability metrics artifact. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day3-build-package-ci-report-baseline.md` | Planned Day 3 build, package, CI, report, benchmark, and support-tier baseline. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day4-epic11-residual-intake.md` | Planned Day 4 Epic 11 residual intake and candidate grouping. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day5-residual-owner-nongoal-map.md` | Planned Day 5 residual owner, dependency, promotion-gate, and non-goal map. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day6-gap-selection-criteria.md` | Planned Day 6 gap-selection scoring rubric and complete-closure criteria. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day7-gap-selection-decision.md` | Planned Day 7 selected-gap decision for Sprints 138-146. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day8-corpus-oracle-evidence-templates.md` | Planned Day 8 corpus, generated-matrix, optional-data, and oracle-row templates. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day9-report-index-freshness-templates.md` | Planned Day 9 report-index and stale-report templates. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day10-package-abi-platform-claim-templates.md` | Planned Day 10 package/ABI, platform-promotion, downstream-proof, and claim-gate templates. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day11-quality-surface-map.md` | Planned Day 11 quality matrix by touched surface. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day12-public-claim-freeze.md` | Planned Day 12 public claim freeze and unsupported-wording audit. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day13-handoff-synthesis.md` | Planned Day 13 Sprint 138 readiness and later-sprint dependency synthesis. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day14-closeout.md` | Planned Day 14 Sprint 137 closeout, residual register, and validation summary. |

## Inherited Input Inventory

| Source | Day 1 reading | Sprint 137 relevance |
| --- | --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 137 | Defines the seven Sprint 137 work items and the 166-hour sprint estimate. | Source of the Day 1 owner map and Sprint 137 deliverable list. |
| `docs/planning/EPIC_12/SPRINT_137/PLAN.md` | Defines the 14-day plan and Day 1 completion criteria. | Controls day-by-day sequencing and validation expectations. |
| `docs/planning/EPIC_12/reviews/review-codex-2026-08-03.md` | States that the project is broad and well validated but still not state of the art. | Provides current gap categories and state-of-the-art proof requirements. |
| `docs/planning/EPIC_12/reviews/todo-codex-2026-08-03.md` | Defines the gap-closure sequence and completion definition. | Provides the execution model for baseline, corpus, QR, partial-SVD, report, runtime, package, platform, adoption, and closeout work. |
| `docs/planning/EPIC_11/EPIC_11_RETROSPECTIVE.md` | Records Epic 11 earned claims, non-claims, and future-epic candidates. | Prevents Day 2-7 work from treating residuals as already-earned claims. |
| `docs/planning/EPIC_11/SPRINT_136/artifacts/day12-residual-queue-publication.md` | Publishes the residual queue, QR residual queue, and explicit non-claim register. | Primary input for Days 4-7 residual reconciliation and gap selection. |

## Initial Residual Families Visible Before Day 2

| Residual family | Current boundary |
| --- | --- |
| QR residual and SuiteSparse/corpus expansion | Evidence- and metadata-blocked; requires distinct trust value, output semantics, tolerance, rank/nullity metadata, optional-data policy, and support tiers before promotion. |
| Partial-SVD edge-case and convergence expansion | Evidence-blocked; requires metrics for singular values, vectors, subspaces, ordering, convergence budget, tolerance, and skip behavior. |
| Corpus/report normalized indexing | Metadata-blocked; row meaning, freshness, support tier, failure class, and claim boundary must be preserved. |
| Runtime/backend sentinels | Evidence-blocked; local measurement only until metrics, tolerances, runtime budget, variance policy, and backend-state semantics are defined. |
| Package/ABI/distribution | Explicit non-claim; shared-library, dynamic ABI, runtime-loader, and package-manager support need product decision and proof. |
| Platform promotion | Evidence-blocked; macOS/Windows install parity and Windows staged tests need CI proof and source portability before promotion. |
| Adoption/documentation maintenance | Optional-local unless tied to earned behavior, package, platform, report, or solver evidence. |

## Initial Validation Expectations

| Change type | Required validation |
| --- | --- |
| Sprint 137 planning artifacts only | `git diff --check` and Epic 12 Markdown link/path validation. |
| Public documentation wording | Documentation hygiene plus claim scan against Epic 11 and Epic 12 non-claims. |
| Script/report-generator edits | Syntax check or focused script execution plus row-meaning and support-tier review. |
| Build/package/CMake/pkg-config edits | Relevant install/export/downstream proof and static/shared support-boundary review. |
| CI workflow edits | Workflow structure review plus hosted-runner support-tier notes. |
| Benchmark/report generation | Capture command, commit, platform, compiler/configuration, row meaning, freshness, support tier, and skip/defer status. |
| `.c` or `.h` edits | `make format && make lint && make test`. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 137 project-plan item has a day-level owner. | Complete | Sprint 137 scope table above and `WORKING_NOTES.md` day-level ownership table. |
| Inherited Epic 11 and Epic 12 inputs are visible before decisions begin. | Complete | Inherited input inventory and residual-family summary above. |
| Validation expectations are documented before later implementation work. | Complete | Validation expectation table above and `WORKING_NOTES.md` validation section. |

