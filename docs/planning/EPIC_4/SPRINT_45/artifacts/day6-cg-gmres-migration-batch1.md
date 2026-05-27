# Sprint 45 Day 6 Artifact: CG / GMRES Migration Batch 1

## Purpose

Use the Day 5 shared iterative workspace layer in the primary scalar repeated-
solve paths that still depended on per-call heap bundles, while keeping Sprint
45 away from block-path and benchmark work.

## Main Day 6 Conclusion

Sprint 45's primary scalar iterative paths are now routed through the reusable
internal workspace seam.

This batch was bounded to:

- matrix-free CG
- matrix-free GMRES, which also covers the matrix-backed `sparse_solve_gmres(...)`
  wrapper because it delegates into `sparse_solve_gmres_mf(...)`

It did **not** widen into:

- block CG
- block GMRES
- MINRES migration
- benchmark or example implementation changes

## Landed Migration Scope

### 1. Matrix-free CG now uses the shared CG workspace view

`sparse_solve_cg_mf(...)` now:

- initializes `sparse_iter_workspace_t`
- prepares `sparse_cg_workspace_view_t`
- binds:
  - `r`
  - `z`
  - `p`
  - `Ap`
- frees the shared owner on every exit path

Interpretation:

- matrix-backed and matrix-free CG now share one internal packed-buffer model
- Sprint 45 no longer has a split CG allocation story between those two paths

### 2. GMRES now uses the shared GMRES workspace view

`sparse_solve_gmres_mf(...)` now:

- initializes `sparse_iter_workspace_t`
- prepares `sparse_gmres_workspace_view_t`
- binds:
  - Arnoldi basis storage
  - Hessenberg storage
  - Givens scratch
  - Hessenberg-space residual/solve vectors
  - the main temporary work vector
- preserves the existing one-shot control flow, restart logic, callback
  behavior, and convergence contract

Because `sparse_solve_gmres(...)` is already a wrapper over
`sparse_solve_gmres_mf(...)`, the matrix-backed GMRES entry point now inherits
the same shared-workspace path automatically.

Interpretation:

- Sprint 45 now covers the main GMRES repeated-allocation hotspot without
  public API churn
- the Day 5 typed prepare seam was sufficient as designed; no redesign was
  needed before the first GMRES landing

## Preserved Boundaries

The batch kept these responsibilities solver-local:

- CG recurrence math
- GMRES restart / Arnoldi control
- callback/progress flow
- stagnation tracking policy
- preconditioner invocation choreography
- true-residual checks and final result aggregation

Interpretation:

- Day 6 migrated storage/view ownership, not algorithm control
- the shared layer remains a narrow allocation/reuse seam rather than a new
  solver framework

## Validation

Because `*.c` files changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Targeted touched-surface follow-ons also passed:

- `./build/test_iterative`
- `./build/test_stagnation`
- `./build/example_matrix_free`

Representative direct rerun outcomes:

- `test_iterative`
  - `test_cg_mf_basic`
  - `test_cg_mf_nos4`
  - `test_gmres_mf_basic`
  - `test_gmres_mf_right_precond`
  - `test_gmres_mf_unsymmetric`
  all passed
- `example_matrix_free`
  - unpreconditioned GMRES converged in `3` iterations
  - diagonally preconditioned GMRES also converged in `3` iterations
  - solution error stayed around `1e-13`

## Sprint 45 Position After Day 6

The next bounded migration order is clearer:

1. audit the post-primary-path state
2. land block-CG as the real multi-RHS workspace target
3. keep block GMRES / MINRES / BiCGSTAB in the wrapper/defer bucket unless a
   later batch stays obviously small
4. add repeated-solve benchmark evidence only after the main internal paths are
   in place

## Bottom Line

Day 6 delivered:

- shared-workspace matrix-free CG
- shared-workspace GMRES
- automatic matrix-backed GMRES participation through the existing wrapper
- a fully green validation baseline for the touched CG/GMRES paths

That is the right bounded primary-path migration batch for Sprint 45.
