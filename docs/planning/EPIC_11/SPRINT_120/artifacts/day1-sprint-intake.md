# Sprint 120 Day 1 Sprint Intake

## Purpose

Establish the Sprint 120 execution baseline before direct/iterative oracle
audits or proof-owner splits begin. Day 1 records authoritative scope,
required inputs, day-level ownership, validation rules, evidence fields, and
non-claim boundaries.

## Authoritative Sprint Scope

Sprint 120 implements the Epic 11 project-plan section:
`Sprint 120: Direct/Iterative Oracle Architecture & Giant-Test Split`.

| Field | Value |
|---|---|
| Duration | 14 days |
| Estimate | 168 hours |
| Goal | Create a maintainable direct/iterative oracle architecture and reduce giant test ownership in the highest-risk direct and iterative proof files. |
| Primary deliverables | Direct/iterative oracle architecture artifact; reduced giant-test ownership; bounded cross-solver oracle pilot; validation and CTest/source-list evidence; residual direct/iterative oracle queue. |

## Project-Plan Item Map

| Item # | Item | Planned Days | Day 1 Intake Notes |
|---:|---|---|---|
| 1 | Oracle Ownership Audit | Days 2-3 | Split into direct and iterative audit days so QR/LDLT/CSC and CG/GMRES/BiCGSTAB/MINRES owners stay visible. |
| 2 | Shared Fixture Design | Days 4-5 | Design must preserve solver-local tolerances, residual interpretation, convergence, failure modes, and callbacks. |
| 3 | Direct Test Split Batch | Days 5-8 | Direct split cannot start until exact file/helper/build boundaries and rollback criteria are recorded. |
| 4 | Iterative Test Split Batch | Days 5, 9-10 | Iterative split cannot hide convergence or progress-callback behavior behind generic fixtures. |
| 5 | Cross-Solver Oracle Pilot | Days 11-12 | Pilot must be bounded to named fixtures and solver paths; no broad parity claim. |
| 6 | Validation | Days 7-8, 10, 12-13 | Focused tests first; full quality required if `.c` or `.h` files change. |
| 7 | Documentation and Closeout | Day 14 | Closeout must publish split/defer outcomes, residuals, validation, and non-claims. |

## Input Artifact Inventory

| Input | Required Use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 120 section | Defines authoritative item scope and estimates. |
| `docs/planning/EPIC_11/SPRINT_120/PLAN.md` | Defines day-by-day execution sequence. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day2-validation-inventory.md` | Provides validation lane expectations and command categories. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day8-product-truth-map.md` | Provides product-truth and public-claim limits. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day9-hotspot-metrics.md` | Provides giant-source and giant-test context. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day10-hotspot-owner-handoff.md` | Provides proof-owner handoff candidates. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day11-evidence-template-design.md` | Provides evidence template structure. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day12-evidence-template-refresh.md` | Provides refreshed template usage expectations. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day13-public-claim-drift-audit.md` | Provides public claim drift checks. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day14-sprint-closeout-handoff.md` | Provides carry-forward framing into Epic 11 implementation sprints. |
| `docs/planning/EPIC_11/SPRINT_118/templates/oracle-expansion-evidence-template.md` | Provides fields for bounded oracle expansion and comparison artifacts. |
| `docs/planning/EPIC_11/SPRINT_118/templates/source-movement-evidence-template.md` | Provides fields for source/test owner movement and rollback artifacts. |
| `docs/planning/EPIC_11/SPRINT_119/artifacts/day13-validation-parity-package.md` | Provides the latest validation packaging pattern. |
| `docs/planning/EPIC_11/SPRINT_119/artifacts/day14-movement-closeout.md` | Provides Sprint 120 handoff rules for proof-owner movement and non-claims. |
| `docs/planning/EPIC_11/SPRINT_119/RETROSPECTIVE.md` | Provides lessons about explicit deferral, focused proof, and claim boundaries. |

## Validation Rules

| Change Type | Required Day-Level Validation |
|---|---|
| Planning documentation only | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_120`. |
| Test-only `.c` or helper `.h` edits | `make format && make lint && make test`; also run focused touched tests before full quality where useful. |
| Source-list or Makefile membership edits | `make source-list-check`; focused build of touched target; full quality if `.c` or `.h` changed. |
| CMake or CTest membership edits | CMake configure/build and `ctest -N` count proof; full quality if `.c` or `.h` changed. |
| Direct solver proof-owner split | Focused direct tests for touched QR, LDLT, LDLT CSC, LU, or Cholesky-adjacent paths, plus required full quality if source/header files changed. |
| Iterative solver proof-owner split | Focused iterative tests for touched CG, GMRES, BiCGSTAB, MINRES, block, preconditioner, or callback paths, plus required full quality if source/header files changed. |
| Cross-solver oracle pilot | Focused pilot test plus adjacent direct/iterative tests that share fixtures or assertions. |
| Public/support wording | Claim scan against Sprint 118 product truth, Sprint 118 public-claim drift audit, and Sprint 119 non-claim register. |

## Non-Claim Boundaries

Sprint 120 may improve proof-owner architecture and validation hygiene. It does
not, by itself, claim:

- broad direct solver parity;
- broad iterative solver parity;
- full external-oracle coverage;
- broad PETSc, Trilinos, SciPy, LAPACK, SuiteSparse, or Eigen parity;
- state-of-the-art solver status;
- portable benchmark or performance superiority;
- package/install support expansion;
- public API expansion;
- symmetric platform validation beyond the lanes actually run.

## Day-Level Owner Map

| Day | Owner Focus | Output |
|---:|---|---|
| 1 | Intake setup | Working notes and intake artifact. |
| 2 | Direct audit | Direct oracle owner inventory. |
| 3 | Iterative audit | Iterative oracle owner inventory. |
| 4 | Shared fixture design | Architecture artifact. |
| 5 | Split ranking | Candidate ranking and proof plan. |
| 6 | Direct split design | Exact direct split checklist. |
| 7 | Direct implementation | Direct split and focused proof. |
| 8 | Direct validation | Direct validation and residual queue. |
| 9 | Iterative split design | Exact iterative split checklist. |
| 10 | Iterative implementation | Iterative split and focused proof. |
| 11 | Pilot design | Bounded cross-solver oracle design. |
| 12 | Pilot implementation | Pilot and focused proof. |
| 13 | Validation package | Source-list, focused, CMake/CTest, and quality evidence. |
| 14 | Closeout | Residuals, non-claims, artifact index, and Sprint 121 handoff. |

## Completion Criteria

| Criterion | Status |
|---|---|
| Every Sprint 120 project-plan item has a day-level owner | Complete |
| Prior evidence templates and source-boundary lessons are identified | Complete |
| Validation requirements are recorded before implementation | Complete |
| Non-claim boundaries are recorded before implementation | Complete |
| No oracle split begins before audit, design, proof, and rollback expectations are recorded | Complete |
