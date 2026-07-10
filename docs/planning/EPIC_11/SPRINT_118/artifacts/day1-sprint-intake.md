# Sprint 118 Day 1 Sprint Intake and Artifact Skeleton

## Purpose

Day 1 establishes the Sprint 118 execution frame. It identifies the required
input artifacts, creates the working-note and artifact structure, maps each
Sprint 118 project-plan item to day-level owners, and records the scope and
validation boundaries that prevent baseline work from becoming premature
Sprint 119-127 implementation work.

## Inputs Reviewed

| Input | Sprint 118 role |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 118 section | Authoritative source for the sprint goal, seven project-plan items, deliverables, and `166` hour estimate. |
| `docs/planning/EPIC_11/SPRINT_118/PLAN.md` | Day-by-day execution plan, daily deliverables, and completion criteria. |
| `docs/planning/EPIC_11/reviews/review-codex-2026-07-09.md` | Current post-Epic-10 review of efficiency, maintainability, usability, documentation, coherence, test coverage, packaging, and state-of-the-art readiness. |
| `docs/planning/EPIC_11/reviews/todo-codex-2026-07-09.md` | Step-by-step Epic 11 closure sequence and claim-discipline rules. |
| `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md` | Final Epic 10 validation anchor, earned claims, explicit non-claims, and carry-forward queue. |
| `docs/planning/EPIC_10/SPRINT_117/RETROSPECTIVE.md` | Final integration sprint result, validation summary, residuals, and Epic 10 closeout handoff. |
| `docs/planning/EPIC_10/SPRINT_117/artifacts/` | Final validation, comparison, claim, residual, and non-claim evidence sources. |
| Prior Epic retrospectives | Deferred-work history used to avoid duplicate residual scheduling. |

## Day 1 Created Structure

| Path | Role |
|---|---|
| `docs/planning/EPIC_11/SPRINT_118/WORKING_NOTES.md` | Running sprint notes, constraints, input inventory, day ownership, validation expectations, and daily updates. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/` | Evidence directory for Sprint 118 day artifacts. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day1-sprint-intake.md` | Day 1 intake artifact and scope/validation contract. |

## Sprint 118 Project-Plan Item Map

| Item | Project-plan item | Day ownership | Expected output |
|---:|---|---|---|
| 1 | Baseline Quality Recheck | Days 1-4 | Intake scope, validation inventory, baseline quality evidence, and CI/platform truth freeze. |
| 2 | Residual Queue Conversion | Days 5-6 | Deduplicated residual intake, owner map, dependencies, proof gates, and deferral notes. |
| 3 | Current Product Truth Map | Days 7-8 | Product-truth design and completed evidence-backed truth map. |
| 4 | Source/Test Hotspot Metrics | Days 9-10 | File-size/responsibility metrics and source/test owner handoff. |
| 5 | Evidence Template Refresh | Days 11-12 | Refreshed evidence templates for source movement, oracle expansion, performance, package/ABI, and adoption cleanup. |
| 6 | Public Claim Drift Audit | Day 13 | Public/support claim audit and cleanup or future-owner recommendations. |
| 7 | Sprint Closeout | Day 14 | Artifact index, validation summary, residual deferred debt, and Sprint 119-127 handoff package. |

## Scope Boundaries

Sprint 118 is a baseline and planning-evidence sprint. It may inventory,
classify, audit, create templates, and publish handoff requirements. It should
not silently perform downstream implementation work.

Out of scope for Day 1 and not allowed without a later explicit scope decision:

- source movement for eigensolver private owners such as `s20_select_indices`,
  `s20_lift_ritz_vectors`, shift-invert setup/conversion, or
  `lanczos_iterate_op`;
- direct/iterative/SVD/QR oracle implementation;
- source-file extraction or giant-test splitting;
- Makefile, CMake, workflow, install, package, benchmark, or ABI behavior
  changes;
- public API expansion;
- package-manager, shared-library, dynamic ABI, GPU, distributed-memory,
  portable-performance, or broad ecosystem-parity claims;
- adoption-surface rewrite work reserved for later Epic 11 sprints.

## Validation Boundary

Day 1 modified only Sprint 118 planning documentation. The required validation
for Day 1 is therefore:

- `git diff --check`;
- focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_118`;
- no `make format && make lint && make test` requirement unless `.c` or `.h`
  files are modified.

Future Sprint 118 days should escalate validation according to the touched
surface:

| Surface | Validation expectation |
|---|---|
| Public/support claim wording | Cross-check against Epic 10 closeout, Sprint 117 artifacts, current product truth, and explicit non-claims. |
| C source or public headers | Run `make format && make lint && make test` and add focused behavior proof where needed. |
| Build, install, package, workflow, or benchmark surfaces | Run the relevant focused reviewed or supplemental lane and record support-tier meaning. |
| Platform support wording | Verify reviewed CI scope, expected CTest counts, staged exclusions, and package/install proof before changing claims. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Sprint 118 working-notes baseline exists. | Complete. |
| Sprint 118 artifact directory exists. | Complete. |
| Input artifact inventory is recorded. | Complete. |
| Every Sprint 118 project-plan item has a day-level owner. | Complete. |
| Scope boundaries and validation expectations are recorded. | Complete. |
| Required Epic 10 and Epic 11 input artifacts are identified. | Complete. |
| Sprint 119-127 implementation work is not silently pulled into Sprint 118 Day 1. | Complete. |
