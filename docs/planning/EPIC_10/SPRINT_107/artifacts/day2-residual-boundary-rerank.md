# Day 2 Residual Boundary Re-rank

## Purpose

Day 2 ranks the Sprint 107 residual source and proof-owner cleanup queue from
live repository evidence. The goal is to avoid duplicating Sprint 106 completed
extractions while preserving the dependency order required by the Sprint 107
plan.

## Inputs

- Sprint 107 Day 1 residual intake:
  `docs/planning/EPIC_10/SPRINT_107/artifacts/day1-residual-debt-intake.md`
- Sprint 106 closeout and handoff:
  `docs/planning/EPIC_10/SPRINT_106/artifacts/day14-validation-closeout.md`
- Sprint 106 retrospective residual deferred debt:
  `docs/planning/EPIC_10/SPRINT_106/RETROSPECTIVE.md`
- Live residual owners:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`
  - `src/sparse_eigs.c`
  - `src/sparse_matrix.c`

## Live Inventory

Approximate live metrics captured from the `sprint-107` branch:

| owner | lines | recent churn count since 2026-06-01 | function-like definitions | assertion/test macro hits | helper/fixture keyword hits |
|---|---:|---:|---:|---:|---:|
| `tests/test_ldlt_csc.c` | 3,884 | 7 | 132 | 588 | 154 |
| `tests/test_qr.c` | 3,234 | 2 | 83 | 411 | 123 |
| `tests/test_svd.c` | 2,879 | 4 | 79 | 464 | 157 |
| `tests/test_iterative.c` | 2,841 | 5 | 89 | 398 | 88 |
| `src/sparse_eigs.c` | 1,538 | 4 | 36 | 0 | 41 |
| `src/sparse_matrix.c` | 1,359 | 9 | 43 | 0 | 13 |
| **total** | **15,735** | | | | |

Interpretation:

- `tests/test_ldlt_csc.c` is the largest and most assertion-heavy residual
  proof owner, so it remains the first extraction target.
- `tests/test_svd.c` has the highest helper/fixture keyword count among the
  residual proof owners, but its rank/oracle semantics make broad extraction
  riskier than QR or iterative builder cleanup.
- `tests/test_iterative.c` has higher recent churn than QR and contains
  convergence-sensitive assertions, so it should start with matrix/RHS
  builders only.
- `src/sparse_matrix.c` has the highest recent churn, but that churn reinforces
  the need for a deferral contract rather than opportunistic extraction because
  it owns central API and compatibility behavior.

## Sprint 106 Duplicate-Work Exclusions

Sprint 106 completed these related maintainability items, so Sprint 107 should
not repeat them:

| completed Sprint 106 work | Sprint 107 exclusion |
|---|---|
| LDLT CSC row-adjacency source extraction into `src/sparse_ldlt_csc_rowadj.c` | Do not revisit LDLT CSC source extraction; Day 3-4 may only extract one test-side proof helper. |
| QR Householder source extraction into `src/sparse_qr_householder.c` and `src/sparse_qr_internal.h` | Do not move more QR implementation code in Sprint 107; Day 5-6 is test fixture cleanup only. |
| LU CSR structural source extraction into `src/sparse_lu_csr_struct.c` and `src/sparse_lu_csr_internal.h` | No LU CSR work is part of Sprint 107 residual debt. |
| graph/reorder fixture extraction into `tests/test_graph_fixtures.h` | Do not spend Sprint 107 capacity on graph/reorder cleanup. |
| direct-solver helper extraction into `tests/test_direct_solver_helpers.h` | Avoid duplicating shared direct-solver helper work; LDLT CSC cleanup must be local and narrow. |
| integration fixture extraction into `tests/test_integration_fixtures.h` | Do not expand integration fixture cleanup unless later code work exposes a direct dependency. |
| source-list and CMake reconciliation | Re-run source-list/CMake checks only when Sprint 107 source ownership changes. |

## Ranked Sprint 107 Cleanup Queue

| rank | owner | Sprint 107 action | rationale | validation cost |
|---:|---|---|---|---|
| 1 | `tests/test_ldlt_csc.c` | Extract one row-adjacency assertion helper or residual/oracle helper after a boundary artifact. | Largest residual proof owner, highest assertion density, explicitly named first in Sprint 106 residual debt. | Focused LDLT CSC test plus full C gate if `.c` edited. |
| 2 | `tests/test_qr.c` | Extract repeated matrix/vector builders while preserving solve and reconstruction assertions inline. | Broad proof fixture owner with lower recent churn, making it a safer first fixture cleanup after LDLT. | Focused QR test plus full C gate if `.c` edited. |
| 3 | `tests/test_iterative.c` | Extract reusable matrix/RHS builders only. | Churn and convergence sensitivity are higher than QR, so helper movement must avoid result/convergence assertions. | Focused iterative tests plus full C gate if `.c` edited. |
| 4 | `tests/test_svd.c` | Perform dedicated cleanup with rank/oracle no-move rules. | Helper density is high, but rank/oracle semantics are sensitive and need a dedicated boundary after simpler fixture patterns. | Focused SVD test plus full C gate if `.c` edited. |
| 5 | `src/sparse_eigs.c` | Create a fresh boundary tied to Sprint 103 comparison surfaces, then split only if a low-risk helper is found. | Source-owner debt is real, but comparison behavior and dispatch/workspace boundaries are more sensitive than test fixture cleanup. | Focused eigensolver tests, source-list check if membership changes, full C gate. |
| 6 | `src/sparse_matrix.c` | Write a central matrix shell deferral contract. | Highest churn but central API/compatibility risk; extraction should wait for explicit public-contract review. | Docs hygiene unless code is unexpectedly touched. |

## Dependency Order

1. Boundary and rank the residual queue before any extraction.
2. Start with LDLT CSC test proof cleanup because it is the largest residual
   proof owner and has the narrowest accepted Sprint 107 instruction.
3. Move to QR and iterative fixture cleanup before SVD because builder-only
   patterns are safer to establish before touching rank/oracle-heavy SVD
   proof surfaces.
4. Treat `src/sparse_eigs.c` after proof-owner cleanup because any source split
   must respect Sprint 103 comparison surfaces and may require build-system
   follow-through.
5. End with `src/sparse_matrix.c` documentation because it is intentionally a
   deferral contract, not a source extraction.

## First-Pass Validation Cost Map

| day range | likely touched surface | required validation |
|---|---|---|
| Days 1-3 | planning artifacts only unless boundary review uncovers urgent doc updates | `git diff --check`; trailing-whitespace scan |
| Day 4 | `tests/test_ldlt_csc.c` or local test helper | focused LDLT CSC test; `make format && make lint && make test` |
| Day 6 | `tests/test_qr.c` or local QR test helper | focused QR test; `make format && make lint && make test` |
| Day 8 | `tests/test_iterative.c` or local iterative test helper | focused iterative test; `make format && make lint && make test` |
| Day 10 | `tests/test_svd.c` or local SVD test helper | focused SVD test; `make format && make lint && make test` |
| Day 12 | optional `src/sparse_eigs.c` source split | focused eigensolver tests; source-list check if a new source file appears; `make format && make lint && make test` |
| Day 13 | matrix shell deferral documentation | `git diff --check`; trailing-whitespace scan |
| Day 14 | mixed closeout | full required checks for all touched file categories |

## Completion Check

- All six Sprint 106 residual items have a Sprint 107 disposition.
- Completed Sprint 106 extractions are excluded from duplicate work.
- Cleanup order is dependency-safe and matches the Sprint 107 day plan.
- Validation cost is explicit before extraction work begins.
