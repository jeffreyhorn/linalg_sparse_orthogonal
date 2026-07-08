# Day 1 Residual Intake and Boundary Refresh

## Purpose

Day 1 converts Sprint 110's residual behavior-owner and proof-owner debt into a
Sprint 113 boundary that downstream days can use without rediscovering scope.
The key constraint is to exclude work already completed in Sprint 110 while
ordering the remaining eigensolver, direct/iterative, and SVD residuals by
dependency.

## Source Evidence Reviewed

| Source | Relevant evidence |
|---|---|
| `docs/planning/EPIC_10/PROJECT_PLAN.md` Sprint 113 | Seven Sprint 113 items and 168-hour project-plan scope. |
| Sprint 110 retrospective | Final residual deferred debt, completed-work exclusions, validation metrics, and non-claims. |
| Sprint 110 Day 7 artifact | Eigensolver behavior-owner candidate map and public handle/workspace selection. |
| Sprint 110 Day 8 artifact | Public handle/workspace validation no-move contract and focused eigensolver validation. |
| Sprint 110 Day 9 artifact | Direct/iterative proof-owner boundary, selected CG exact-RHS cleanup, and remaining candidates. |
| Sprint 110 Day 11 artifact | SVD proof-loop boundary, selected rank-deficient 5x4 setup helper, and remaining candidates. |
| Sprint 111 retrospective | User-facing docs do not depend on unstable internal source ownership. |
| Sprint 112 retrospective | Package/platform support truth is static-first and must not be widened by Sprint 113 work. |

## Duplicate-Work Exclusion Fence

The following work is explicitly excluded from Sprint 113 unresolved debt:

| Excluded work | Why excluded |
|---|---|
| Matrix builder ownership decision | Completed in Sprint 110 before Matrix Market movement. |
| Matrix builder private source implementation | Completed in `src/sparse_matrix_build_internal.c`. |
| Matrix Market private source split | Completed in `src/sparse_matrix_io.c` with public declarations unchanged. |
| Matrix Market focused validation and solver-smoke checks | Completed in Sprint 110 validation artifacts. |
| Public eigensolver handle/workspace validation | Completed as a no-move contract in Sprint 110 Day 8. |
| Iterative CG exact-RHS allocation/setup cleanup | Completed in Sprint 110 Day 10. |
| SVD rank-deficient 5x4 setup cleanup | Completed in Sprint 110 Day 12. |
| Sprint 112 package/platform proof | Complete baseline; Sprint 113 must not reinterpret it as ABI/platform expansion. |

## Remaining Eigensolver Behavior-Owner Candidates

| Candidate | Primary owner area | Dependencies before movement | Initial Day 1 ordering |
|---|---|---|---:|
| Defaults and option validation | public options and entry validation in `src/sparse_eigs.c` / `include/sparse_eigs.h` | Direct tests for `opts == NULL`, invalid options, backend enums, and error codes. | 1 |
| Backend dispatch | backend selection and execution paths in `src/sparse_eigs.c` | Direct tests for AUTO priority, `backend_used`, backend result propagation, and error propagation. | 2 |
| Grow-m sizing and retry behavior | grow-m backend and workspace capacity paths | Direct tests for retry growth, progress callbacks, partial results, peak basis, and residuals. | 3 |
| Refinement defaults and budgets | refinement helpers and backend return boundaries | Direct tests for mutation of returned eigenpairs, cancellation boundaries, and backend status preservation. | 4 |
| Shift-invert setup | nearest-sigma and shift-invert operator setup | Direct tests for singular shifts, LDLT path reporting, and inverse Ritz conversion. | 5 |
| Shared Lanczos kernels | cross-backend Lanczos helpers | Direct tests for ordering, residual scale, reorthogonalization, and vector lifting across backends. | 6 |
| Public handle/workspace source movement | public handle and workspace bridge | Existing validation no-move contract must be superseded by broader direct proof. | 7 |

Day 2 should compare the first six candidates first. Public handle/workspace
source movement should not be selected unless Day 2 finds a narrower
non-duplicate proof target that goes beyond Sprint 110's completed validation.

## Remaining Direct/Iterative Proof-Owner Candidates

| Candidate | Primary file | Dependency / risk | Initial Day 1 ordering |
|---|---|---|---:|
| LDLT CSC external dense-reference oracle cleanup | `tests/test_ldlt_csc.c` | Needs dedicated oracle-lane review because dense comparison, Windows skip behavior, permutation handling, and residual checks are coupled. | 1 |
| GMRES exact-RHS setup | `tests/test_iterative.c` | Family-specific restart, convergence, residual, and lucky-breakdown behavior must remain visible. | 2 |
| BiCGSTAB exact-RHS setup | `tests/test_iterative.c` | Breakdown and residual behavior are solver-specific and should not be hidden by a cross-solver helper. | 3 |
| MINRES exact-RHS setup | `tests/test_iterative.c` | Symmetry/preconditioner assumptions and residual behavior must remain visible. | 4 |
| CG preconditioner-specific exact-RHS setup | `tests/test_iterative.c` | Must avoid repeating Sprint 110's generic CG allocation/setup helper. | 5 |
| QR sequential RHS setup | `tests/test_qr.c` | Must avoid repeating Sprint 109 QR exact-RHS helper work; literals often explain least-squares/refinement proof. | 6 |

Day 7 should still make the final selection from evidence, but Day 1 orders the
oracle and non-CG iterative families earlier because they are not the completed
Sprint 110 cleanup path.

## Remaining SVD Proof-Owner Candidates

| Candidate | Primary file | Dependency / risk | Initial Day 1 ordering |
|---|---|---|---:|
| Dense low-rank proof-loop cleanup | `tests/test_svd.c` | Repeated proof loops may be cleanup candidates if retained singular-value and Frobenius residual evidence stays visible. | 1 |
| Sparse low-rank proof-loop cleanup | `tests/test_svd.c` | Dense-vs-sparse residuals, drop tolerance, and corpus fixture names must stay visible. | 2 |
| Partial-SVD vector/residual cleanup | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | Vector orthogonality and `A*v ~= sigma*u` residuals are claim-critical. | 3 |
| Moore-Penrose helper extraction | `tests/test_svd.c` | Expected inverse entries and product dimensions must remain visible. | 4 |
| Reconstruction helper movement | `tests/test_svd.c` | Reconstruction residuals and matrix layout loops are claim-critical. | 5 |
| U/Vt orthogonality helper movement | `tests/test_svd.c` | Dot-product orthogonality thresholds and layout loops are claim-critical. | 6 |
| Condition-number proof cleanup | `tests/test_svd.c` | Expected finite/infinite condition values and rectangular interpretation must remain visible. | 7 |

Day 9 must refresh this boundary before any SVD cleanup. Day 1 does not approve
movement of any SVD proof loop.

## Dependency Order for Sprint 113

1. Residual intake and duplicate-work exclusion must precede all selection
   work.
2. Eigensolver owner selection must precede eigensolver proof design.
3. Eigensolver proof design must precede tests or source movement.
4. Eigensolver proof must precede movement/no-move decision.
5. Direct/iterative proof-owner boundary must precede cleanup.
6. SVD proof-boundary refresh must precede cleanup.
7. Metrics and non-claims must follow implementation and cleanup work.
8. Integrated validation planning must precede final validation execution.
9. Final closeout must follow passing required checks.

## Day 1 Completion Criteria

- Completed Sprint 110 work is fenced off from Sprint 113 unresolved debt.
- Remaining eigensolver behavior-owner candidates are inventoried and ordered.
- Remaining direct/iterative proof-owner candidates are inventoried and
  ordered.
- Remaining SVD proof-owner candidates are inventoried and ordered.
- Sprint 113 working notes and artifact directory exist.
- Downstream days can proceed without rediscovering residual scope.
