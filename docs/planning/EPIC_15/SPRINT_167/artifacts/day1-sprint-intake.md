# Sprint 167 Day 1: Sprint Intake And Artifact Setup

## Purpose

Day 1 establishes the Sprint 167 artifact package, ties the sprint to the
active Epic 15 project plan, and defines the initial evidence categories that
will drive the residual audit, evidence ledger, CI/report inventory, and claim
gate work across the rest of the sprint.

## Source Plan

The active source plan is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section "Sprint 167: Epic 15 Baseline,
Evidence Ledger & Claim Gate".

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`; that is treated
as a stale path for this sprint. The current branch follows the merged Epic 15
project plan.

## Sprint 167 Scope

Sprint 167 establishes the Epic 15 baseline by:

- extracting unresolved residuals from Epic 13 and Epic 14;
- inventorying source, header, test, script, generated report, package,
  install, documentation, and CI surfaces;
- creating an Epic 15 evidence ledger that separates supported, partially
  supported, local-only, hosted-only, advisory, deferred, and unsupported
  claims;
- selecting the exact Epic 15 gaps that future sprints will close;
- defining acceptance criteria, validation commands, and stop conditions for
  each selected gap;
- leaving a clear handoff for Sprint 168 hosted performance publication work.

## Inputs Reviewed

| Input | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_15/PROJECT_PLAN.md` | Authoritative sprint scope, item list, estimates, deliverables, and total. |
| `docs/planning/EPIC_15/SPRINT_167/PLAN.md` | Day-by-day execution plan and Day 1 completion criteria. |
| `docs/planning/EPIC_15/reviews/review-codex-2026-08-18.md` | Review findings and highest-value gaps for Epic 15. |
| `docs/planning/EPIC_15/reviews/todo-codex-2026-08-18.md` | Step-by-step closure sequence for Epic 15 gaps. |

## Artifact Structure

| Path | Purpose |
| --- | --- |
| `docs/planning/EPIC_15/SPRINT_167/PLAN.md` | Sprint 167 day-by-day plan. |
| `docs/planning/EPIC_15/SPRINT_167/WORKING_NOTES.md` | Rolling sprint notes, assumptions, non-goals, status labels, and daily log. |
| `docs/planning/EPIC_15/SPRINT_167/artifacts/day1-sprint-intake.md` | Day 1 sprint intake and baseline artifact. |
| `docs/planning/EPIC_15/SPRINT_167/artifacts/` | Destination for Days 2-14 evidence, inventory, ledger, selection, gate, and closeout artifacts. |

## Initial Evidence Category Map

| Category | Day 1 definition | Claim boundary |
| --- | --- | --- |
| Claims | Current user-facing capability, support, package, report, and validation statements. | A claim needs source-controlled evidence and a validation owner before being treated as supported. |
| Non-claims | Explicitly unsupported, deferred, or scoped-out capabilities. | Non-claims must stay visible when evidence is absent or intentionally narrow. |
| Reports | Generated, normalized, source-controlled, local-only, hosted, and advisory report outputs. | Report rows must not imply hosted or release evidence unless CI owns that proof. |
| CI | Hosted Linux, macOS, Windows, reviewed, and supplemental workflow evidence. | Hosted proof must be mapped to exact workflow lanes and not generalized across platforms. |
| Package | Static-first install, CMake package, pkg-config metadata, ABI, shared-library, and package-manager surfaces. | Static archive proof must not imply shared-library, dynamic ABI, runtime-loader, or package-manager support. |
| API | Public headers, generated API docs, examples, tutorials, and declaration coherence. | Documentation cleanup must not imply behavior changes unless implementation and tests change. |
| Platform | Platform-specific build, test, install, report, and package support. | Platform claims must name Linux, macOS, Windows, local-only, hosted-only, reviewed, or supplemental status. |
| Performance | Benchmarks, methodology metadata, sentinels, report freshness, and comparison limits. | Performance rows must remain methodology-bound and must not become broad superiority claims. |
| Comparison | External oracle/comparison fixtures, tolerances, generated rows, and freshness checks. | Comparison claims must remain bounded by solver family, fixture family, comparator, and tolerance policy. |
| Failure behavior | Allocation failure, cleanup invariants, OOM behavior, and partial-construction handling. | Failure-path claims need targeted tests and explicit subsystem scope. |

## Evidence Status Labels

| Label | Definition |
| --- | --- |
| Supported | Evidence exists and has a matching local validation command or reviewed hosted CI lane. |
| Partially supported | Evidence exists but is narrower than a broad user-facing interpretation. |
| Hosted-only | Evidence depends on a hosted workflow and is not locally reproduced during the sprint. |
| Local-only | Evidence is locally reproducible but not hosted, published, or release-grade. |
| Advisory | Evidence supports planning, navigation, or interpretation but not a hard product claim. |
| Deferred | The project intentionally postpones the capability and must document the reason. |
| Unsupported | The project must not claim the capability. |

## Initial Non-Goal Register

| Non-goal | Reason |
| --- | --- |
| Broad state-of-the-art claim | Epic 15 must continue evidence-bound claim discipline rather than extrapolating from selected proof. |
| Broad external-library parity | Existing comparison work is bounded by selected solver and fixture families. |
| Portable performance superiority | Current benchmark evidence is methodology-bound and not broad superiority proof. |
| Shared-library support | Static-first package behavior remains the supported package contract until an explicit product decision changes it. |
| Dynamic ABI stability | Package version metadata is not a dynamic ABI compatibility promise. |
| Runtime-loader behavior | No shared-library loader behavior is currently selected or proven. |
| Package-manager distribution | Source install and package metadata proof are not package-manager distribution support. |
| Broad platform parity | Linux, macOS, and Windows evidence levels remain tiered unless future lanes prove parity. |
| Hosted generated API publication | Local generated API HTML is not hosted proof unless a publication lane is added. |
| Source/header behavior changes | Sprint 167 Day 1 is documentation and planning artifact setup only. |

## Stop-Condition Register

Stop and revise if any Sprint 167 work:

1. promotes fixture-scoped solver comparison into broad correctness or
   external-library parity;
2. promotes local-only generated reports into hosted or release evidence;
3. promotes methodology-bound benchmark rows into portable performance
   superiority;
4. treats static-first package proof as shared-library, dynamic ABI,
   runtime-loader, package-manager, or broad platform package support;
5. edits public claim wording before the evidence ledger has an owner row;
6. changes `.c` or `.h` files without running
   `make format && make lint && make test`;
7. leaves generated build, install, coverage, cache, or report output in the
   intended commit.

## Day 2 Handoff

Day 2 should start the prior-epic residual audit by reviewing:

- `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md`;
- `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`;
- any residual or non-claim sections that identify work carried into Epic 15.

The Day 2 output should separate resolved residuals from still-open residuals
before Day 3 risk and value classification.

## Validation Notes

Day 1 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 167 scope is tied to the active Epic 15 project plan. | Complete | Source plan path and stale prompt path are recorded in this artifact and `WORKING_NOTES.md`. |
| Artifact locations are created and named consistently. | Complete | `WORKING_NOTES.md` and `artifacts/day1-sprint-intake.md` exist under `docs/planning/EPIC_15/SPRINT_167/`. |
| Evidence categories are defined before audit work begins. | Complete | Initial evidence category map and status labels are defined above. |
