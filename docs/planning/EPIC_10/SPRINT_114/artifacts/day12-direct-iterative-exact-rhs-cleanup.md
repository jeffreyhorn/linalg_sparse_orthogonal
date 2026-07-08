# Sprint 114 Day 12: Direct/Iterative Exact-RHS Cleanup

## Purpose

Day 12 implements the bounded direct/iterative exact-RHS cleanup designed on
Day 11. The cleanup reduces repeated allocation/fill/matvec boilerplate while
keeping solver-specific proof values visible at the tests: solver options,
preconditioner setup, restart sizes, convergence status, iteration
comparisons, and residual thresholds.

## Implemented Cleanup

| Area | File | Change |
|---|---|---|
| QR | `tests/test_qr.c` | No code movement was needed. The existing `make_qr_exact_rhs` helper already owned the bounded sequential RHS pattern and focused QR validation stayed green. |
| CG | `tests/test_iterative.c` | Reused the existing file-local `require_iterative_exact_rhs` helper for generated sequential RHS setup in `test_cg_diagonal_preconditioner` and `test_cg_precond_laplacian`. |
| GMRES | `tests/test_iterative.c` | Reused `require_iterative_exact_rhs` for generated sequential/sine RHS setup in `test_gmres_large_unsymmetric`, `test_gmres_max_iter_exceeded`, `test_gmres_restart_comparison`, and `test_gmres_diagonal_preconditioner`. |
| BiCGSTAB | `tests/test_bicgstab.c` | Added file-local `make_bicgstab_sequential_rhs` and `require_bicgstab_sequential_rhs`, then used them in SuiteSparse and cross-solver sequential RHS tests. |
| MINRES | `tests/test_minres.c` | Added file-local exact-RHS pattern helpers for sequential, sine, and scaled sequential vectors, then used them in bounded SPD/KKT/preconditioner/direct-comparison tests. |

## Proof Values Preserved

- QR rank, mode, reorder, true residual, and refinement assertions remain at
  their existing call sites.
- CG diagonal preconditioner arrays, 2D Laplacian fixture, iteration
  comparison, and residual thresholds remain visible.
- GMRES restart sizes, tolerance values, nonconvergence expectations,
  preconditioner choices, and true residual checks remain visible.
- BiCGSTAB ILU/ILUT options, difficult-corpus accepted nonconvergence branch,
  convergence assertions, and residual thresholds remain visible.
- MINRES SPD/KKT fixture construction, IC/Jacobi/ILU preconditioners,
  MINRES-vs-CG/LDLT comparisons, iteration counts, and residual gates remain
  visible.

## Non-Claims

- No cross-solver exact-RHS oracle was introduced.
- No helper moved out of its owning test translation unit.
- No public API, install header, source-list, Make, CMake, helper target, or
  reviewed CTest membership changed.
- No QR proof behavior changed; QR was validated as part of the direct solver
  scope but did not need additional helper movement.

## Focused Validation

Focused validation passed:

```text
make build/test_qr && ./build/test_qr && ./build/test_iterative && ./build/test_bicgstab && ./build/test_minres
```

Observed focused summaries:

- `test_qr`: `73` tests, `0` failures, `654` assertions.
- `test_iterative`: `80` tests, `0` failures, `713` assertions.
- `test_bicgstab`: `61` tests, `0` failures, `464` assertions.
- `test_minres`: `43` tests, `0` failures, `702` assertions.

## Required Full Gate

Day 12 modifies `.c` tests, so the required final gate is:

```text
make format && make lint && make test
```

## Completion Criteria

- Solver-specific proof values remain visible at call sites.
- Focused direct/iterative tests pass.
- No broad direct/iterative proof abstraction is claimed.
- Full quality gate is required before the day is complete.
