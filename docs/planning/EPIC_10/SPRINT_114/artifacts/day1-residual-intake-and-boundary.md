# Day 1 Residual Intake and Duplicate Fence

## Purpose

Day 1 turns Sprint 113's residual deferred debt into a Sprint 114 execution
boundary. The main outcome is a duplicate fence: completed Sprint 113 proof
work remains evidence, not unresolved debt. The remaining work is ordered so
that no source movement, helper movement, or broad proof abstraction depends on
evidence that comes later.

## Source Evidence Reviewed

| Source | Relevant evidence |
|---|---|
| `docs/planning/EPIC_10/PROJECT_PLAN.md` Sprint 114 | Ten Sprint 114 items and 168-hour project-plan scope. |
| `docs/planning/EPIC_10/SPRINT_114/PLAN.md` Day 1 | Required Day 1 tasks, deliverables, and completion criteria. |
| Sprint 113 retrospective | Residual deferred debt and explicit non-claims carried into Sprint 114. |
| Sprint 113 Day 4 artifact | Grow-m behavior proof pattern and completed test batch. |
| Sprint 113 Day 5 artifact | Eigensolver movement/no-move decision, including proof requirements for future movement. |
| Sprint 113 Day 6 artifact | Published eigensolver no-move contract. |
| Sprint 113 Day 8 artifact | LDLT CSC external dense-reference oracle cleanup pattern. |
| Sprint 113 Day 10 artifact | Partial-SVD residual helper cleanup pattern. |
| Sprint 113 Day 11 artifact | Proof-owner metrics and non-claims. |
| Sprint 113 Day 14 artifact | Closeout and dependency-ordered residual handoff. |

## Duplicate-Work Exclusion Fence

The following work is explicitly excluded from Sprint 114 unresolved debt:

| Excluded work | Why excluded | Sprint 114 handling |
|---|---|---|
| Sprint 113 residual intake and boundary refresh | Completed and summarized in Sprint 113 working notes and retrospective. | Use as background only. |
| Grow-m behavior owner selection and design | Completed before Sprint 113 implementation. | Use as proof design pattern. |
| Grow-m behavior test batch | Completed in Sprint 113 `tests/test_eigs.c` work. | Do not duplicate; add different eigensolver proofs. |
| Eigensolver movement/no-move decision | Completed in Sprint 113 Day 5. | Revisit only after Sprint 114 prerequisite proofs. |
| Eigensolver no-move contract | Completed in Sprint 113 Day 6. | Preserve unless Day 10 earns a narrow movement. |
| LDLT CSC external dense-reference oracle cleanup | Completed in Sprint 113 Day 8. | Use as cleanup example; do not repeat. |
| Partial-SVD vector residual cleanup | Completed in Sprint 113 Day 10 and review fixes. | Use as helper strictness example; do not repeat. |
| Proof-owner metrics and non-claims artifact | Completed in Sprint 113 Day 11. | Extend metrics only after new work lands. |
| Integrated validation planning and execution | Completed in Sprint 113 Days 12-13. | Reuse validation pattern. |
| Sprint 113 closeout and handoff | Completed in Sprint 113 Day 14 and retrospective. | Use as Sprint 114 residual source. |

## Remaining Eigensolver Proof Owners

| Remaining owner | Primary files / tests | Dependency before movement | Sprint 114 order |
|---|---|---|---:|
| `lanczos_iterate_op` behavior across dispatch paths | `src/sparse_eigs.c`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | Must prove basic, thick-restart, and LOBPCG-adjacent observable behavior. | 1 |
| Repeated/clustered Ritz selection | Ritz selection logic and eigensolver tests | Must prove repeated and clustered spectrum behavior before moving Ritz selection. | 2 |
| Ritz vector lifting and publication boundary | vector lifting and public result publication | Must prove residual/normalization and requested/converged publication shape before helper extraction. | 3 |
| Partial-result publication after `m_cap` exhaustion | bounded grow-m / Lanczos result publication | Must prove converged count, result shape, and non-overrun behavior. | 4 |
| Shift-invert grow-m conversion | nearest-sigma and shift-invert eigensolver paths | Must prove conversion and public result invariants before source ownership changes. | 5 |
| Eigensolver source movement decision | source ownership and build metadata | Must follow the preceding proof owners; otherwise continue no-move contract. | 6 |

This order intentionally places all proof work before the Day 10 movement
decision. A broad eigensolver source split remains out of scope.

## Remaining Direct/Iterative Proof Owners

| Remaining owner | Primary file | Proof values at risk | Sprint 114 order |
|---|---|---|---:|
| QR sequential RHS setup | `tests/test_qr.c` | literal RHS values, least-squares residuals, refinement before/after residuals. | 1 |
| CG preconditioner-specific exact-RHS setup | `tests/test_iterative.c` | preconditioner setup, residual norms, iteration comparisons. | 2 |
| GMRES exact-RHS setup | `tests/test_iterative.c` | restart settings, convergence status, residual norms, lucky-breakdown behavior. | 3 |
| BiCGSTAB exact-RHS setup | `tests/test_iterative.c` | breakdown behavior, convergence status, residual norms. | 4 |
| MINRES exact-RHS setup | `tests/test_iterative.c` | symmetry assumptions, preconditioner behavior, residual norms. | 5 |

The direct/iterative cleanup must remain solver-specific. A broad oracle
abstraction remains blocked until more solver-specific lanes prove a common
owner.

## Remaining SVD Proof Owners

| Remaining owner | Primary file | Proof values at risk | Sprint 114 order |
|---|---|---|---:|
| Reconstruction helper movement by storage contract | `tests/test_svd.c` | reconstruction residuals, dimensions, storage layout. | 1 |
| U/Vt orthogonality helper movement by leading dimension | `tests/test_svd.c` | economy/full leading dimensions and dot-product thresholds. | 2 |
| Moore-Penrose product helper extraction | `tests/test_svd.c` | product dimensions and Moore-Penrose identities. | 3 |
| Dense low-rank proof-loop cleanup | `tests/test_svd.c` | singular-value error bounds and Frobenius residuals. | 4 |
| Sparse low-rank proof-loop cleanup | `tests/test_svd.c` | dense-vs-sparse residuals, drop tolerance, corpus fixture names. | 5 |
| Condition-number proof logic cleanup | `tests/test_svd.c` | finite/infinite condition values and rectangular interpretation. | 6 |

The SVD cleanup batch is Day 13 work, after eigensolver movement truth and
direct/iterative cleanup are known. A broad SVD proof abstraction remains out
of scope.

## Dependency Order for Sprint 114

1. Residual intake and duplicate-work exclusion must precede all proof work.
2. `lanczos_iterate_op` behavior design must precede its implementation.
3. `lanczos_iterate_op` proof must precede Ritz selection proof work.
4. Repeated/clustered Ritz selection proof must precede any Ritz selection
   movement.
5. Ritz vector lifting and publication-boundary proof must precede any shared
   vector-publication helper extraction.
6. Partial-result publication proof must precede final shift-invert grow-m
   conversion proof.
7. Shift-invert grow-m conversion proof must precede eigensolver source
   movement decisions.
8. Eigensolver movement or continued no-move decision must precede
   direct/iterative cleanup handoff.
9. Direct/iterative exact-RHS cleanup must preserve solver-specific proof
   values and avoid broad abstraction claims.
10. SVD cleanup must preserve storage, leading-dimension, product-dimension,
    low-rank, and condition-number proof values.
11. Final validation, metrics, and non-claim handoff must follow all touched
    implementation work.

## Day-Level Project-Plan Ownership

| Project item | Sprint 114 owner days | Day 1 disposition |
|---:|---|---|
| 1 | Day 1 | Completed by this artifact and working notes baseline. |
| 2 | Days 2-3 | Ready for Lanczos behavior proof design and implementation. |
| 3 | Days 4-5 | Blocked until Lanczos proof lands. |
| 4 | Days 6-7 | Blocked until Ritz selection proof lands. |
| 5 | Day 8 | Blocked until vector publication proof lands. |
| 6 | Day 9 | Blocked until partial-result proof lands. |
| 7 | Day 10 | Blocked until all eigensolver proof prerequisites land. |
| 8 | Days 11-12 | Blocked until eigensolver movement/no-move truth is known. |
| 9 | Day 13 | Blocked until direct/iterative cleanup completes. |
| 10 | Day 14 | Blocked until all implementation and focused validation work completes. |

## Day 1 Completion Criteria

- Completed Sprint 113 work is fenced off from Sprint 114 unresolved debt.
- Remaining eigensolver proof owners are inventoried and dependency ordered.
- Remaining direct/iterative proof owners are inventoried and dependency
  ordered.
- Remaining SVD proof owners are inventoried and dependency ordered.
- Sprint 114 working notes and artifact directory exist.
- Downstream days can proceed without rediscovering residual scope.
