# Day 8 Iterative Fixture Cleanup

## Purpose

Day 8 implements the iterative fixture cleanup selected by the Day 7 boundary.
The change reduces repeated matrix-free test setup in `tests/test_iterative.c`
without moving solver options, convergence checks, residual checks,
preconditioner setup, or direct comparison assertions out of the tests.

## Implemented Cleanup

### Local RHS Helper

Added one local static helper near the existing iterative test helpers:

```c
static void fill_sequential_rhs(double *b, idx_t n);
```

The helper only fills `b[i] = (double)(i + 1)`. It does not allocate memory,
configure a solver, run a solver, compute residuals, or assert convergence.

### Matrix Builder Reuse

Replaced repeated hand-built matrix setup in the two matrix-free basic tests:

| call site | replacement |
|---|---|
| `test_cg_mf_basic` | `build_spd_tridiag(n, 4.0, -1.0)` |
| `test_gmres_mf_basic` | `build_unsym_tridiag(n, 5.0, -2.0, -1.0)` |

The GMRES replacement preserves the existing unsymmetric fixture because
`build_unsym_tridiag` takes arguments in `diag, upper, lower` order.

### RHS Fill Call Sites

Replaced the selected literal sequential RHS loops with `fill_sequential_rhs`
in:

- `test_cg_mf_basic`
- `test_cg_mf_nos4`
- `test_gmres_mf_basic`
- `test_gmres_mf_right_precond`

## Proof Preservation

The cleanup keeps the following proof logic inline at the test sites:

- CG and GMRES option structs;
- matrix-free callback selection;
- SuiteSparse matrix load checks;
- ILU factorization and teardown;
- convergence assertions;
- iteration-count equality checks;
- matrix-free versus concrete-solver solution comparisons;
- maximum-difference assertion for the `nos4` matrix-free comparison.

No exact-solution allocation, restart logic, preconditioner behavior, or
residual assertion was extracted.

## Build-System and Registration Impact

- No public header changes.
- No internal header changes.
- No Makefile or CMake changes.
- No new compiled helper target.
- No `RUN_TEST` additions, removals, or renames.
- No reviewed CTest registration count change is expected.

## Validation Plan

Because Day 8 edits a `.c` file, validation is:

```sh
make build/test_iterative && ./build/test_iterative
make format && make lint && make test
git diff --check
```

Also run a trailing-whitespace scan over Sprint 107 planning docs, the Epic 10
project plan, and touched C files.

## Deferred Iterative Cleanup

The following cleanup remains deferred:

- exact-solution plus RHS allocation helpers across broader CG/GMRES tests;
- SuiteSparse load plus exact-RHS helper extraction;
- restart comparison setup helpers;
- diagonal, ILU, and right-preconditioner setup helpers;
- diagonal matrix-free operator builders;
- helpers that include convergence, residual, iteration-count, or
  direct-solver comparison assertions.
