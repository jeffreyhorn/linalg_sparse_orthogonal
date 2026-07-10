# Sprint 119 Day 1 Sprint Intake And Evidence Setup

## Purpose

Day 1 establishes the Sprint 119 execution frame before any eigensolver source
movement begins. The sprint goal is to convert the safest eigensolver residual
movements into validated source-boundary improvements without widening public
claims.

## Sprint 119 Inputs

| Input | Required use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 119 section | Authoritative item list, estimates, goal, prerequisites, and deliverables. |
| `docs/planning/EPIC_11/SPRINT_119/PLAN.md` | Day-by-day execution plan. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day6-residual-owner-map.md` | Residual owner table, dependency order, and proof-gate checklist. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day8-product-truth-map.md` | Eigensolver baseline truth, candidate claims, and explicit non-claims. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day10-hotspot-owner-handoff.md` | Eigensolver source/test handoff, source-movement prerequisites, and no-move/defer guidance. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day12-evidence-template-refresh.md` | Refreshed template set and future-sprint usage rules. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day14-sprint-closeout-handoff.md` | Sprint 119 first proof gate and residual deferred debt handoff. |
| `docs/planning/EPIC_11/SPRINT_118/templates/source-movement-evidence-template.md` | Reusable template fields for movement, validation, drift, non-claims, and residual handoff. |

## Project-Plan Item Map

| Item # | Item name | Estimate | Sprint 119 day owners |
|---:|---|---:|---|
| 1 | Movement Feasibility Audit | 18 hours | Days 2-3 |
| 2 | Source Boundary Design | 26 hours | Days 4-5 |
| 3 | First Movement Batch | 30 hours | Days 5-7 |
| 4 | Selection/Lifting Batch | 34 hours | Days 8-10 |
| 5 | Shift-Invert Boundary Decision | 24 hours | Days 11-12 |
| 6 | Validation and Parity | 20 hours | Days 7, 10, 12-13 |
| 7 | Closeout and Non-Claims | 16 hours | Day 14 |

## Day-Level Execution Map

| Day | Focus | Primary output |
|---:|---|---|
| 1 | Intake and evidence setup | Working notes, input inventory, validation boundaries. |
| 2 | Movement candidate inventory | Candidate and consumer map. |
| 3 | Feasibility ranking | First movement recommendation and defer conditions. |
| 4 | Source boundary design | Old/new file, internal header, build impact, rollback design. |
| 5 | Focused consumer proof design | Source-movement evidence draft and focused test plan. |
| 6 | First movement implementation | Lowest-risk movement batch. |
| 7 | First movement validation | Focused proof, CTest count, source-list/CMake, and quality evidence. |
| 8 | Selection/lifting proof audit | Move/defer decision for `s20_select_indices` and `s20_lift_ritz_vectors`. |
| 9 | Selection/lifting movement or deferral | Implement cleared movement or publish explicit residual. |
| 10 | Selection/lifting validation | Grow-m, thick-restart, and LOBPCG-adjacent proof. |
| 11 | Shift-invert decision | Split/defer decision with LDLT lifecycle and cleanup proof. |
| 12 | Shift-invert validation | Movement or deferral validation. |
| 13 | Full validation package | Source-list, Make/CMake, CTest count, focused eigensolver, and quality evidence. |
| 14 | Closeout | Movement summary, residuals, non-claims, and Sprint 120 handoff. |

## Validation Boundary

| Touched surface | Required validation |
|---|---|
| Documentation-only planning artifacts | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_119`. |
| `.c` or `.h` files | `make format && make lint && make test`. |
| Source-list or Makefile membership | `make source-list-check` or relevant reviewed wrapper. |
| CMake source/test membership | CMake configure/build and `ctest -N` count proof as affected. |
| Eigensolver behavior | Focused tests for affected grow-m, thick-restart, LOBPCG, shift-invert, and repeated-handle paths. |
| Public support wording | Claim-boundary check against Sprint 118 Day 8 product truth and explicit non-claims. |

## Source-Movement Gate

No eigensolver movement should proceed until the owner artifact records:

1. behavior boundary;
2. exact old/new files;
3. internal header and private API contract;
4. source-list, Makefile, and CMake impact;
5. expected CTest count before and after;
6. focused consumer proof;
7. rollback or defer plan;
8. public API and public-claim impact;
9. explicit non-claims preserved.

## Non-Claim Boundary

Sprint 119 may improve internal maintainability and proof ownership, but it
must not claim:

- broad eigensolver parity;
- ARPACK, SciPy, LAPACK, PETSc, Trilinos, or ecosystem parity;
- nonsymmetric eigensolver support;
- portable eigensolver performance superiority;
- state-of-the-art eigensolver replacement status;
- platform, package, ABI, benchmark, or adoption improvements not directly
  proven by this sprint.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Sprint 119 working-notes baseline exists. | Complete. |
| Artifact directory structure exists. | Complete. |
| Day-level owner map is recorded. | Complete. |
| Sprint 118 input artifact inventory is recorded. | Complete. |
| Validation and non-claim boundary notes are recorded. | Complete. |
| Every Sprint 119 project-plan item has a day-level owner. | Complete. |
| No movement begins before feasibility, proof, and rollback expectations are recorded. | Complete. |
