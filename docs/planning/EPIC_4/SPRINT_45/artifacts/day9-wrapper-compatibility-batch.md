# Sprint 45 Day 9 Artifact: Wrapper Compatibility Batch

## Purpose

Normalize the remaining one-shot block iterative wrapper layer so it reads
explicitly as a compatibility/composition surface over the existing scalar
solver entries, without widening Sprint 45 into another solver or workspace
redesign.

## Main Day 9 Conclusion

Sprint 45's block wrapper paths now share one explicit internal compatibility
helper instead of each re-implementing the same per-column delegation and
aggregation shape.

This batch was bounded to:

- block GMRES wrapper normalization
- block MINRES wrapper normalization
- block BiCGSTAB wrapper normalization
- preservation of current one-shot public behavior

It did **not** widen into:

- scalar solver algorithm changes
- new workspace models
- MINRES workspace migration
- benchmark/example work
- public API changes

## Landed Compatibility Scope

### 1. The block wrappers now share one internal per-column delegation helper

Day 9 added one small internal helper that owns:

- the per-column solve loop
- scalar-solver delegation
- aggregate result collection
- convergence/stagnation/breakdown rollup
- first hard-error propagation

The following wrappers now route through that shared helper:

- `sparse_gmres_solve_block(...)`
- `sparse_minres_solve_block(...)`
- `sparse_bicgstab_solve_block(...)`

Interpretation:

- the wrapper/composition contract is now explicit in code
- Sprint 45 reduced wrapper duplication without touching the underlying solver
  math

### 2. The scalar entries remain the behavioral truth

After Day 9, the block wrappers clearly behave as convenience layers over:

- `sparse_solve_gmres(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_bicgstab(...)`

Interpretation:

- the scalar entries remain the solver-behavior owner
- the block wrappers remain compatibility-oriented composition surfaces rather
  than independent solver implementations

## Preserved Boundaries

The batch kept these responsibilities outside the new helper:

- solver iteration logic
- recurrence/state math
- workspace ownership inside the scalar solver entries
- preconditioner behavior
- benchmark/reporting policy

Interpretation:

- Day 9 normalized wrapper structure, not solver internals
- Sprint 45 stayed away from another broad workspace migration batch

## Validation

Because `*.c` files changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Targeted touched-wrapper follow-ons also passed:

- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`

Representative direct rerun outcomes:

- `test_block_solvers`
  - all `15` tests passed
- `test_minres`
  - all `43` tests passed
- `test_bicgstab`
  - all `58` tests passed

## Sprint 45 Position After Day 9

The remaining sprint order is now clearer:

1. repeated-solve benchmark design/evidence
2. optional later MINRES extension only if it stays small

Interpretation:

- the direct workspace migration queue is already closed
- the wrapper/composition cleanup queue is now substantially reduced
- Sprint 45 is ready to pivot into measured repeated-solve evidence

## Bottom Line

Day 9 delivered:

- explicit shared wrapper compatibility logic for block GMRES/MINRES/BiCGSTAB
- preserved one-shot public behavior
- a green full validation baseline
- direct touched-wrapper confirmation on the three affected test binaries

That is the right bounded compatibility batch for Sprint 45.
