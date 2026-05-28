# Sprint 45 Day 8 Artifact: Block Iterative Migration Batch

## Purpose

Use the already-landed shared iterative workspace seam in the one real
remaining direct multi-RHS workspace target from Day 7:
`sparse_cg_solve_block(...)`.

Keep the batch bounded away from block GMRES / MINRES / BiCGSTAB churn,
benchmark work, and public API changes.

## Main Day 8 Conclusion

Sprint 45's true block CG path now participates in the same reusable internal
workspace model as the primary scalar CG and GMRES paths.

This batch was bounded to:

- block CG
- adoption of `sparse_block_cg_workspace_view_t`
- preservation of the current one-shot public behavior and algorithm choices

It did **not** widen into:

- block GMRES
- block MINRES
- block BiCGSTAB
- MINRES scalar migration
- repeated-solve benchmark work
- examples or public API changes

## Landed Migration Scope

### 1. Block CG now uses the shared block-CG workspace view

`sparse_cg_solve_block(...)` now:

- initializes `sparse_iter_workspace_t`
- prepares `sparse_block_cg_workspace_view_t`
- binds:
  - `R`
  - `Z`
  - `P`
  - `AP`
  - `bnorms`
  - `rz`
  - `conv`
  - `rnorms`
- frees the shared owner on every touched return path

Interpretation:

- block CG no longer owns its own direct per-call heap bundle
- the Day 5 shared owner/view seam now spans:
  - scalar CG
  - matrix-free CG
  - matrix-free + matrix-backed GMRES
  - block CG

### 2. The migration used the existing shared model instead of inventing a block-only one

Before Day 8, `sparse_cg_solve_block(...)` allocated:

- four packed `n * nrhs` vector bundles
- `bnorms`
- `rz`
- `conv`
- `rnorms`

Day 8 replaced that with the existing shared owner and typed block view rather
than adding another specialized block allocator.

Interpretation:

- Sprint 45 still has one coherent iterative workspace design
- the block path now reuses the same ownership/capacity/slice model as the
  earlier scalar migrations

## Preserved Boundaries

The batch kept these responsibilities solver-local:

- block-CG recurrence math
- per-column convergence checks
- preconditioner invocation choreography
- shared block SpMV usage
- final result aggregation and reporting

Interpretation:

- Day 8 migrated storage/view ownership, not algorithm control
- the shared layer remains a narrow allocation/reuse seam rather than a new
  block-solver framework

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
- `./build/test_block_solvers`

Representative direct rerun outcomes:

- `test_block_solvers`
  - all `15` tests passed
  - `test_block_cg_iteration_count` remained:
    - `block_cg iters=17`
    - `single_cg iters=17`
- `test_iterative`
  - all visible scalar CG / GMRES / matrix-free iterative cases passed

## Sprint 45 Position After Day 8

The remaining sprint order is now clearer:

1. wrapper/composition review
2. repeated-solve benchmark evidence
3. optional later MINRES extension only if it stays small

Interpretation:

- the direct workspace-migration queue is now substantially complete
- later Sprint 45 work should focus on compatibility clarity and efficiency
  proof rather than new core storage seams

## Bottom Line

Day 8 delivered:

- shared-workspace block CG
- preservation of the current one-shot public behavior
- a green full validation baseline
- direct touched-binary confirmation on both iterative and block-solver
  surfaces

That is the right bounded multi-RHS migration batch for Sprint 45.
