# Day 1 Residual Debt Intake

## Purpose

Day 1 converts the Sprint 106 residual deferred debt into a bounded Sprint 107
starting package. The goal is not to solve the debt immediately. The goal is to
make the residual owners, constraints, validation expectations, and Day 2
ranking inputs explicit before any helper or source extraction begins.

## Authoritative Inputs

- `docs/planning/EPIC_10/PROJECT_PLAN.md`
  - Sprint 107 section: "Residual Maintainability Debt & Proof-Owner Cleanup"
- `docs/planning/EPIC_10/SPRINT_107/PLAN.md`
  - Day-by-day implementation plan
- `docs/planning/EPIC_10/SPRINT_106/RETROSPECTIVE.md`
  - "Residual Deferred Debt" section
- `docs/planning/EPIC_10/SPRINT_106/WORKING_NOTES.md`
  - validation rules, extraction constraints, and completed work context
- `docs/planning/EPIC_10/SPRINT_106/artifacts/`
  - completed extraction, fixture, source-list, and closeout artifacts

## Sprint 106 Carry-Forward Debt

The Sprint 106 retrospective carries forward six work items:

| owner | carry-forward instruction | Sprint 107 disposition |
|---|---|---|
| `tests/test_ldlt_csc.c` | Remains the largest direct CSC proof owner; extract only one row-adjacency assertion or residual/oracle helper after a narrow boundary artifact. | Days 2-4 own boundary ranking and one bounded helper extraction. |
| `tests/test_qr.c` | Still has broad proof fixtures; extract repeated matrix/vector builders without hiding solve and reconstruction intent. | Days 5-6 own QR fixture boundary and cleanup. |
| `tests/test_iterative.c` | Needs convergence-sensitive helper cleanup; start with reusable matrix/RHS builders only. | Days 7-8 own iterative builder boundary and cleanup. |
| `tests/test_svd.c` | Needs dedicated SVD proof-owner cleanup with focused validation before moving rank/oracle helpers. | Days 9-10 own SVD boundary and bounded cleanup. |
| `src/sparse_eigs.c` | Remains tied to Sprint 103 comparison surfaces and needs a fresh boundary before splitting workspace or dispatch helpers. | Days 11-12 own eigensolver boundary and split-or-deferral. |
| `src/sparse_matrix.c` | Remains central API/compatibility territory and should not be split opportunistically. | Day 13 owns the central matrix shell deferral contract. |

## Current Residual Owner Sizes

Line counts captured from the live `sprint-107` branch:

| owner | current lines | Day 1 interpretation |
|---|---:|---|
| `tests/test_ldlt_csc.c` | 3,884 | Largest residual proof owner; first bounded extraction target. |
| `tests/test_qr.c` | 3,234 | Broad proof fixture owner; cleanup must preserve reconstruction and solve intent. |
| `tests/test_svd.c` | 2,879 | SVD proof owner; rank/oracle interpretation needs focused validation before movement. |
| `tests/test_iterative.c` | 2,841 | Convergence-sensitive proof owner; start only with reusable matrix/RHS builders. |
| `src/sparse_eigs.c` | 1,538 | Source-owner candidate tied to Sprint 103 evidence; boundary required before split. |
| `src/sparse_matrix.c` | 1,359 | Central API/compatibility owner; document deferral rather than splitting now. |
| **total** | **15,735** | Requires ordered, bounded cleanup rather than broad rewrite. |

## Explicit Constraints

Sprint 107 inherits these constraints from Sprint 106:

- no public API or install-header change;
- no new compiled test helper target;
- no reviewed test-count change unless explicitly approved;
- no broad direct-solver rewrite;
- no broad QR, iterative, eigensolver, or SVD proof-owner cleanup;
- no central sparse matrix shell extraction.

## Workstream Inventory

| workstream | project-plan item | planned days | Day 1 owner status |
|---|---|---:|---|
| residual boundary re-rank | Item 1 | Days 1-2 | Ready for Day 2 ranking. |
| LDLT CSC proof-owner follow-through | Item 2 | Days 3-4 | Needs boundary before edit. |
| QR fixture cleanup | Item 3 partial | Days 5-6 | Needs builder-only boundary. |
| iterative fixture cleanup | Item 3 partial | Days 7-8 | Needs convergence-sensitive boundary. |
| SVD proof-owner cleanup | Item 4 | Days 9-10 | Needs rank/oracle no-move rules. |
| eigensolver boundary and first split | Item 5 | Days 11-12 | Needs Sprint 103 evidence-aware source boundary. |
| central matrix shell deferral contract | Item 6 | Day 13 | Documentation-only unless new evidence requires more. |
| validation and closeout | Item 7 | Day 14 | Full gate required if any `.c` or `.h` changed. |

## Validation Expectations

| touched surface | required validation |
|---|---|
| planning documentation only | `git diff --check`; trailing-whitespace scan on touched Sprint 107 planning files |
| test-only `.c` cleanup | focused affected test binary; `make format && make lint && make test` |
| source `.c` cleanup | focused affected tests; source-list check if membership changes; `make format && make lint && make test` |
| header cleanup | focused affected tests; source-list/build checks if ownership changes; `make format && make lint && make test` |
| build-system update | source-list checker; focused Make/CMake validation; full code gate if C files are also touched |
| docs plus code cleanup | docs hygiene plus the full code gate required by touched code |

## Day 2 Starting Questions

Day 2 should answer:

1. Which residual owner has the highest review-value cleanup after accounting
   for validation cost?
2. Which candidate helper movement would duplicate work already completed in
   Sprint 106 and should therefore be excluded?
3. Which call-site proof intent must remain inline for QR, iterative, and SVD
   tests?
4. Does `src/sparse_eigs.c` have a genuinely low-risk first split, or should
   Sprint 107 document deferral after boundary review?
5. What concrete preconditions would make future `src/sparse_matrix.c`
   extraction safe enough for a later sprint?

## Day 1 Exit Criteria

- Sprint 107 artifacts directory exists.
- Working notes contain validation rules and cleanup boundaries.
- All Sprint 106 residual debt is mapped to a Sprint 107 workstream.
- Validation expectations are explicit before boundary or extraction work
  starts.
