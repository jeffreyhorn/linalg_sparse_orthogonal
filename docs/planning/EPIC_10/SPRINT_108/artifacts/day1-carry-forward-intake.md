# Day 1 Carry-Forward Intake

## Purpose

Day 1 converts the Sprint 108 project-plan section and Sprint 107 residual
deferred debt into an actionable, bounded work package. The main risk is
accidentally repeating Sprint 107 cleanup or treating deliberately deferred
source-owner work as permission for opportunistic extraction. This artifact
records the Sprint 108 owner inventory, exclusions, day-level ownership, and
validation expectations before any boundary or code cleanup begins.

## Source Inputs

- `docs/planning/EPIC_10/PROJECT_PLAN.md`, Sprint 108 section.
- `docs/planning/EPIC_10/SPRINT_107/RETROSPECTIVE.md`, residual deferred debt.
- `docs/planning/EPIC_10/SPRINT_107/WORKING_NOTES.md`.
- Sprint 107 closeout and source-boundary artifacts:
  - `day12-eigensolver-source-deferral.md`
  - `day13-central-matrix-shell-deferral-contract.md`
  - `day14-validation-metrics-closeout.md`

## Sprint 108 Workstream Inventory

| Workstream | Owner | Carry-Forward Work | Explicit Guardrail |
|---|---|---|---|
| Residual proof-owner boundary refresh | four large test owners | Re-rank remaining test proof-owner debt after Sprint 107. | Exclude completed Sprint 107 helper extractions. |
| LDLT CSC oracle follow-through | `tests/test_ldlt_csc.c` | Extract at most one additional named assertion, residual, or oracle helper. | Keep direct CSC proof intent and failure localization visible. |
| QR fixture follow-through | `tests/test_qr.c` | Consider generated fixtures, tall/economy builders, diagonal/singleton setup, and SuiteSparse exact-RHS setup. | Keep solve, rank, reconstruction, refinement, and residual assertions visible. |
| Iterative convergence cleanup | `tests/test_iterative.c` | Consider repeated convergence-sensitive setup. | Do not hide solver options, restarts, preconditioners, convergence results, or direct comparisons. |
| SVD oracle/reconstruction cleanup | `tests/test_svd.c` | Consider rank, oracle, reconstruction, pseudoinverse, low-rank, partial-SVD, and condition-number proof logic. | Create a dedicated validation lane before moving any helper family. |
| Eigensolver source feasibility | `src/sparse_eigs.c` | Plan dense Jacobi or grow-m refinement source boundaries. | No source split before Make/CMake/source-list and cross-backend spectral validation are planned. |
| Matrix shell public-behavior review | `src/sparse_matrix.c` | Review public behavior and private-header dependencies. | No central shell extraction before compatibility and public-behavior guardrails exist. |
| Validation and closeout | touched sprint surfaces | Match validation strength to touched files. | No accidental public API, install-header, helper-target, or reviewed test-count drift. |

## Completed Sprint 107 Work Excluded From Sprint 108

These items are already completed and must not be reintroduced as Sprint 108
scope:

- residual intake from Sprint 106;
- residual owner re-rank from Sprint 107;
- LDLT CSC row-adjacency helper extraction;
- QR small 4x3 fixture-builder cleanup;
- iterative matrix-free fixture cleanup;
- SVD diagonal and rank-1 fixture cleanup;
- eigensolver no-split source boundary and deferral record;
- central matrix shell deferral contract;
- Sprint 107 validation and drift checks.

## Day-Level Ownership

| Day | Focus | Sprint 108 Item(s) | Primary Output |
|---:|---|---|---|
| 1 | Scope and carry-forward intake | 1, 8 | Working notes and intake artifact. |
| 2 | Residual proof-owner boundary refresh | 1 | Ranked cleanup boundary with exclusions. |
| 3 | LDLT CSC oracle boundary | 2 | LDLT CSC helper boundary or deferral. |
| 4 | LDLT CSC helper follow-through | 2 | Bounded helper change or no-change record. |
| 5 | QR residual fixture boundary | 3 | QR cleanup candidate list and guardrails. |
| 6 | QR fixture follow-through | 3 | Bounded QR fixture cleanup. |
| 7 | Iterative convergence boundary | 4 | Iterative cleanup candidate list and guardrails. |
| 8 | Iterative convergence cleanup | 4 | Bounded iterative cleanup. |
| 9 | SVD validation lane boundary | 5 | SVD validation lane and helper candidate. |
| 10 | SVD proof cleanup | 5 | Bounded SVD helper-family cleanup. |
| 11 | Eigensolver source feasibility boundary | 6 | Source feasibility and validation map. |
| 12 | Eigensolver feasibility closeout | 6 | Future extraction checklist or no-change rationale. |
| 13 | Matrix shell public-behavior review | 7 | Public-behavior and private-header dependency review. |
| 14 | Validation, metrics, and closeout | 8 | Final validation, metrics, and residual queue. |

## Validation Expectations

| Scenario | Required Validation |
|---|---|
| Documentation-only day | `git diff --check` and trailing-whitespace scan over touched docs. |
| Test `.c` day | Focused touched test suite plus `make format && make lint && make test`. |
| Test helper/header day | Focused impacted tests plus `make format && make lint && make test`. |
| Implementation source day | Focused family tests, source-list/build checks if membership changes, and `make format && make lint && make test`. |
| Build-system day | Make, CMake, source-list parity, and full quality gate. |
| Public header/install day | Public API/install/export review, downstream/package checks as applicable, and full quality gate. |
| Mixed day | Apply the strongest requirement from any touched surface. |

## Initial Non-Goals

- No public API change.
- No install-header change.
- No new compiled helper target.
- No reviewed test-count change unless explicitly planned and reviewed.
- No broad solver-family redesign from fixture cleanup.
- No `src/sparse_eigs.c` source split without a feasibility and validation
  plan.
- No `src/sparse_matrix.c` shell extraction without public-behavior and
  private-header dependency review.

## Completion Criteria Status

- Every Sprint 108 project-plan item has day-level ownership.
- Sprint 107 completed extractions are listed as exclusions.
- Validation expectations are explicit before boundary or cleanup work starts.

