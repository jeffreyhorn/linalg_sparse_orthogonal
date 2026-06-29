# Sprint 96 Day 9: Solver/Algorithm Source Cleanup Batch 2

## Purpose

Day 9 closes the selected solver/algorithm cleanup lane after the Day 8
iterative block-solver source split. The closeout reconciles current ownership
comments and records validation coverage for the finished solver cleanup.

## Code Closeout Summary

Updated current-ownership comments in:

- `src/sparse_iterative_block.c`
- `src/sparse_iterative.c`
- `src/sparse_iterative_internal.h`

The comments now make these ownership boundaries explicit:

- `src/sparse_iterative_block.c` owns the multiple-RHS public entry points and
  per-column adapter glue
- scalar CG, GMRES, MINRES, and BiCGStab remain in their existing scalar
  owners
- `s85_iter_cg_defaults(...)` is shared so block CG uses the same default
  options as scalar CG
- shared result/default helpers remain private to source files under `src/`

No solver behavior changed on Day 9.

## Completed Solver Boundary

The Sprint 96 solver cleanup is complete for the selected scope:

- `sparse_cg_solve_block(...)` now lives in `src/sparse_iterative_block.c`
- `sparse_gmres_solve_block(...)` now lives in `src/sparse_iterative_block.c`
- `sparse_minres_solve_block(...)` now lives in `src/sparse_iterative_block.c`
- `sparse_bicgstab_solve_block(...)` now lives in
  `src/sparse_iterative_block.c`
- public declarations remain in `include/sparse_iterative.h`
- `src/sparse_iterative.c` remains the scalar CG, GMRES, and BiCGStab owner
- `src/sparse_iterative_minres.c` remains the scalar MINRES owner
- Makefile and CMake both register `src/sparse_iterative_block.c`

## Explicit Non-Changes

Day 9 did not change:

- public APIs
- public headers
- option or result structs
- solver algorithms
- convergence criteria
- tests
- benchmarks
- generated documentation

## Stale-Reference Checks

Required stale-reference checks for the solver split:

```sh
rg -n "sparse_.*solve_block|solve_block_|iter_block_column_solver" src include tests
rg -n "sparse_iterative_block|sparse_iterative.c" Makefile CMakeLists.txt src
rg -n "block-solver|block solver|right-hand side" src/sparse_iterative.c src/sparse_iterative_block.c src/sparse_iterative_internal.h
```

Expected state after closeout:

- public block declarations are still in `include/sparse_iterative.h`
- block implementations are in `src/sparse_iterative_block.c`
- build registration exists in both Makefile and CMake
- current comments explain ownership rather than sprint chronology

## Validation

Day 9 touched `.c` and `.h` files, so the full required code-day quality chain
was run:

```sh
make format && make lint && make test
```

The chain passed.

Relevant solver proof owners passed inside the full test run:

- `test_block_solvers`
- `test_minres`
- `test_bicgstab`
- `test_iterative`
- `test_sprint10_integration`
- `test_sprint13_integration`

The lint build compiled benchmark and example binaries without executing them.

## Solver Residual Queue

No additional solver behavior work is required for Sprint 96.

Deferred items remain intentionally outside the Sprint 96 solver cleanup:

- scalar solver algorithm rewrites
- iterative workspace implementation changes
- public API restructuring
- QR, eigensolver, SVD, LU, LDLT, or Cholesky ownership cleanup
- benchmark-driver restructuring
- proof-owner test splitting, which belongs to the later giant-test lane

## Exit State

The selected solver/algorithm source cleanup is complete and validated. Sprint
96 can move to the giant-test architecture lane with the direct-family and
solver-family implementation cleanup lanes closed for the selected scope.
