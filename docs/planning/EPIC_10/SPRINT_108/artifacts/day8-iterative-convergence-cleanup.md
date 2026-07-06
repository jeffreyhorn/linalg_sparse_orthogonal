# Day 8 Iterative Convergence Cleanup

## Purpose

Day 8 applies the Day 7 approved iterative cleanup without hiding the solver
behavior that makes the GMRES diagonal-preconditioner tests useful. The cleanup
is limited to repeated fixture construction in `tests/test_iterative.c`.

## Implemented Helper

Added one local helper near the existing iterative matrix builders:

```c
static SparseMatrix *build_scaled_unsym_tridiag_with_diag_inv(idx_t n,
                                                              double diag_base,
                                                              double diag_step,
                                                              double upper,
                                                              double lower,
                                                              double **diag_inv_out);
```

The helper owns only setup for the repeated poorly scaled unsymmetric
tridiagonal fixture:

- creates the sparse matrix;
- computes `d = diag_base + diag_step * i` for each row;
- inserts diagonal, lower-neighbor, and upper-neighbor entries;
- allocates and fills the matching diagonal-inverse vector;
- frees partial fixture state on allocation or insert failure.

## Updated Call Sites

Only the approved Day 8 call sites were changed:

- `test_gmres_right_precond_diag`
- `test_gmres_diagonal_preconditioner`

Both call sites still show:

- the problem size;
- exact RHS setup;
- `diag_precond_t` construction;
- `sparse_gmres_opts_t` literals;
- restart and tolerance values;
- `precond_side` selection where applicable;
- unpreconditioned, left-preconditioned, and right-preconditioned solve calls;
- convergence assertions;
- iteration reporting;
- true residual and reported residual checks.

## Explicitly Preserved Surfaces

No changes were made to:

- CG convergence/tolerance/max-iteration tests;
- GMRES restart comparison tests;
- SuiteSparse corpus loading or direct comparison tests;
- matrix-free callback and matrix-vs-matrix-free tests;
- public headers;
- implementation sources;
- Makefile or CMake membership;
- CTest registration or reviewed test counts.

## Metrics

| Metric | Before Day 8 | After Day 8 |
|---|---:|---:|
| `tests/test_iterative.c` lines | 2,828 | 2,849 |
| Local scaled unsymmetric tridiagonal + diagonal-inverse builder | no | yes |
| Approved call sites using shared fixture builder | 0 | 2 |
| New compiled helper target | 0 | 0 |
| Public headers touched | 0 | 0 |

The line count increased because the extracted fixture now checks allocation
and sparse insertion failures and frees partial state before returning `NULL`.
The behavioral proof remains at the call sites.

## Residual Iterative Proof-Owner Debt

Deferred iterative cleanup remains intentionally bounded:

- CG convergence lanes should keep solver options, residuals, and direct
  comparison assertions visible until a separate CG-specific boundary exists.
- GMRES restart lanes should keep restart values and convergence outcomes
  visible until a restart-specific cleanup boundary exists.
- SuiteSparse corpus lanes should keep matrix loading, RHS generation,
  tolerance/restart choices, residual checks, and oracle comparisons colocated.
- Matrix-free lanes should keep callback/operator setup and direct matrix parity
  assertions visible because Sprint 107 already completed the safe shared setup.

## Completion Criteria Status

- The approved helper was implemented locally.
- Only the approved call sites were updated.
- Solver options, preconditioner semantics, convergence checks, and residual
  comparisons remain visible.
- No helper target, public API, build membership, or CTest surface changed.
- Focused and full quality validation are required because `tests/test_iterative.c`
  changed.
