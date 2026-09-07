# Day 1 Closeout Intake and Evidence Source Map

## Purpose

Day 1 establishes the final-validation closeout frame for the requested Sprint
197 branch and records the source artifacts, claim surfaces, validation gates,
risks, and open questions that later days must use before editing claims or
status.

## Scope Trace

| Final-validation item | Day 1 owner surface | Intake result |
| --- | --- | --- |
| 206.1 Evidence Reconciliation | Evidence ledger and artifact inventory | Scaffolded in `WORKING_NOTES.md`; future Sprint 198-205 artifacts are explicitly missing at Day 1. |
| 206.2 Claim Recalibration | Public, maintainer, API, benchmark, corpus, and planning docs | Claim surfaces inventoried; edits deferred until evidence reconciliation exists. |
| 206.3 Project Plan Status | Epic 18 project-plan item status | Deferred until prior sprint outcomes and evidence links are known. |
| 206.4 Integrated Validation | Local gates and hosted evidence | Initial validation matrix created with full C gate trigger rules. |
| 206.5 Epic Retrospective | Epic 18 retrospective structure | Structure identified from Epic 17 precedent; content deferred until outcomes are known. |
| 206.6 Residual Queue | Next-epic residual queue | Seed categories identified; no closure status assigned prematurely. |

## Current Evidence Inventory

| Artifact | Exists on Day 1 | Use in later closeout |
| --- | --- | --- |
| `docs/planning/EPIC_17/EPIC_17_RETROSPECTIVE.md` | Yes | Historical baseline and residual context. |
| `docs/planning/EPIC_17/EPIC_17_RESIDUAL_QUEUE.md` | Yes | Prior residual source for Epic 18 selection validation. |
| `docs/planning/EPIC_18/reviews/review-codex-2026-09-04.md` | Yes | Current review findings and state-of-the-art assessment baseline. |
| `docs/planning/EPIC_18/reviews/todo-codex-2026-09-04.md` | Yes | Step-by-step closure inputs. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Yes | Sprint 197-206 plan, estimates, prerequisites, and final-validation item owners. |
| `docs/planning/EPIC_18/SPRINT_197/PLAN.md` | Yes | Requested day-by-day final-validation plan. |
| `docs/planning/EPIC_18/SPRINT_198` through `SPRINT_205` | No | Future evidence; must not be treated as completed or available. |
| Hosted CI and PR review records for Epic 18 implementation sprints | No sprint-specific records yet | Later closeout must link exact run URLs/comment IDs when evidence exists. |

## Claim Surfaces To Reconcile Later

| Surface | Claim type to watch |
| --- | --- |
| `README.md` | Top-level support, platform, package, quality, benchmark, and state-of-the-art wording. |
| `INSTALL.md` | Install support tiers, package-manager support, static-first behavior, CMake/pkg-config guidance. |
| `docs/maintainer_guide.md` | Validation ownership, hosted evidence interpretation, and non-claim boundaries. |
| `benchmarks/README.md` | Methodology-bound benchmark evidence and non-portable performance wording. |
| `docs/api_reference.md` | Generated API publication and source-of-truth semantics. |
| `include/*.h` | Public API/header claims; any edit triggers full C validation. |
| `docs/solver_selection.md`, `docs/cookbook.md`, `docs/tutorial.md` | Adoption, diagnostics, and workflow guidance. |
| `tests/corpus/README.md`, `tests/corpus/schemas/report_index_fields.md` | Report-index, selected comparison, and freshness semantics. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Final sprint and item status. |

## Validation Gate Seed

| Surface changed later | Minimum gate family |
| --- | --- |
| Documentation only | `git diff --check`, `make docs-check`, plus focused claim guards when relevant. |
| `.c` or `.h` source/header files | `make format`, `make lint`, `make test`, and focused gates for the changed owner. |
| CMake/install/package metadata | CMake install/export checks, package/static guards, docs checks, and full C gate if code/header files changed. |
| Windows workflow or PowerShell evidence | Windows PowerShell guard/validate targets and hosted Windows CI evidence. |
| Report index or selected comparison metadata | Report freshness/normalizer/manifest checks plus docs checks. |
| Benchmark evidence | Benchmark freshness tests, selected benchmark report checks, and methodology documentation checks. |

## Initial Risks

| Risk | Day 1 handling |
| --- | --- |
| Sprint numbering mismatch between requested `SPRINT_197` path and project-plan final-validation Sprint 206 section. | Recorded explicitly in working notes; item references use 206.1-206.6 for traceability. |
| Future Sprint 198-205 artifacts are not yet present. | Marked as missing evidence, not as failed or complete evidence. |
| Claim recalibration could happen before evidence reconciliation. | Public/maintainer/API edits deferred until Day 2 and later ledgers exist. |
| Local checks could be mistaken for hosted platform proof. | Validation map separates local gates from hosted CI evidence. |
| Documentation-only Day 1 work could accidentally imply support changes. | Day 1 changes are limited to planning notes and artifact creation. |

## Day 1 Completion Criteria

- Every final-validation item has an owner artifact or validation source.
- Current Epic 18 evidence sources are listed.
- Missing future evidence is recorded before claim edits begin.
- Claim surfaces and validation gate families are mapped.
- No code, header, README, INSTALL, maintainer-guide, or support-claim edits
  were made.

