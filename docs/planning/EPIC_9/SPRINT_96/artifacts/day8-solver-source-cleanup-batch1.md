# Sprint 96 Day 8: Solver/Algorithm Source Cleanup Batch 1

## Purpose

Day 8 lands the first bounded solver/algorithm source cleanup selected on Day
7. The implementation split moves iterative block-solver ownership out of
`src/sparse_iterative.c` and into `src/sparse_iterative_block.c` without
changing public APIs or numerical behavior.

## Implementation Summary

Created the new source owner:

- `src/sparse_iterative_block.c`

Moved these implementations from `src/sparse_iterative.c`:

- `sparse_cg_solve_block(...)`
- `iter_block_column_solver_fn`
- `solve_block_independent_columns(...)`
- `solve_block_gmres_column(...)`
- `sparse_gmres_solve_block(...)`
- `solve_block_minres_column(...)`
- `sparse_minres_solve_block(...)`
- `solve_block_bicgstab_column(...)`
- `sparse_bicgstab_solve_block(...)`

Kept these scalar solver owners unchanged:

- scalar CG in `src/sparse_iterative.c`
- scalar GMRES and matrix-free GMRES in `src/sparse_iterative.c`
- scalar BiCGStab and matrix-free BiCGStab in `src/sparse_iterative.c`
- scalar MINRES in `src/sparse_iterative_minres.c`

## Internal Contract Changes

`src/sparse_iterative_internal.h` now exposes the minimum private helpers
needed by the split block owner:

- `s85_iter_cg_defaults(...)`
- `s85_iter_result_reset(...)`
- `s85_iter_result_mark_converged(...)`

The CG default accessor preserves the existing scalar CG default values for
block CG when `opts == NULL` without duplicating those values in the new file.
The result helpers preserve existing zero-initialization and converged-state
semantics across the source split.

No public header declarations changed.

## Build Registration

Registered `src/sparse_iterative_block.c` in:

- `Makefile`
- `CMakeLists.txt`

The new source is listed next to `src/sparse_iterative.c` and
`src/sparse_iterative_minres.c`.

## Behavior Preservation

The move preserves the existing behavior:

- block CG still uses the block workspace and block SpMV path
- block GMRES still dispatches one right-hand side at a time through
  `sparse_solve_gmres(...)`
- block MINRES still dispatches one right-hand side at a time through
  `sparse_solve_minres(...)`
- block BiCGStab still dispatches one right-hand side at a time through
  `sparse_solve_bicgstab(...)`
- public block solver signatures in `include/sparse_iterative.h` are unchanged
- tests and benchmarks are unchanged

## Stale-Reference Scans

Ran:

```sh
rg -n "sparse_.*solve_block|solve_block_|iter_block_column_solver" src include tests
rg -n "sparse_iterative_block|sparse_iterative.c" Makefile CMakeLists.txt src
rg -n "block-solver|block solver|right-hand side" src/sparse_iterative.c src/sparse_iterative_block.c src/sparse_iterative_internal.h
```

Results:

- public block solver declarations remain in `include/sparse_iterative.h`
- block solver implementations now live in `src/sparse_iterative_block.c`
- scalar iterative implementations remain in existing scalar owners
- Makefile and CMake both register `src/sparse_iterative_block.c`
- no stale block-solver ownership comment was found in the touched iterative
  source files

## Validation

Required code-day quality chain:

```sh
make format
make lint
make test
```

All three passed.

Relevant proof owners passed inside the full test run:

- `test_block_solvers`
- `test_minres`
- `test_bicgstab`
- `test_iterative`
- `test_sprint10_integration`
- `test_sprint13_integration`

The lint build also compiled benchmark and example binaries without executing
them, including iterative benchmark owners.

## Day 9 Follow-Up

Day 9 should stay in the same solver/algorithm lane and focus on closeout:

- inspect `src/sparse_iterative.c`, `src/sparse_iterative_block.c`, and
  `src/sparse_iterative_internal.h` for ownership comments worth clarifying
- decide whether the private helper names need additional internal-header
  comments
- avoid public API changes and algorithm rewrites
- rerun the full required quality chain after any `.c` or `.h` touch

No behavioral follow-up is required from Day 8.
