# Day 7 Iterative Fixture Boundary

## Purpose

Day 7 defines the safe cleanup boundary for `tests/test_iterative.c` before
any iterative solver test code is edited. The Day 8 cleanup should reduce
repeated matrix and RHS setup while keeping convergence behavior, residual
expectations, preconditioner behavior, restart policy, and direct solver
comparison proof intent visible at the test sites.

## File Snapshot

- File: `tests/test_iterative.c`
- Current size: 2,841 lines
- Existing local builders and helpers:
  - `build_spd_tridiag`
  - `build_identity`
  - `build_laplacian_2d`
  - `build_unsym_tridiag`
  - `compute_rhs`
  - matrix-free callback helpers near the matrix-free test sections

The file already has enough local helper surface for a first cleanup batch. Day
8 should not add a shared helper header, compiled helper target, public header,
Makefile entry, CMake entry, or test registration change.

## Repetition Inventory

### Safe Matrix Builders

These candidates are fixture construction and can be moved or reused without
hiding solver assertions:

| candidate pattern | representative tests | disposition |
|---|---|---|
| SPD tridiagonal matrix with diagonal 4 and off-diagonal -1 | many CG tests, `test_cg_mf_basic` | already has `build_spd_tridiag`; safe to reuse in matrix-free basic test |
| unsymmetric tridiagonal matrix with named diagonal, upper, and lower values | GMRES tests, `test_gmres_mf_basic` | already has `build_unsym_tridiag`; safe to reuse in matrix-free basic test |
| 2D Laplacian matrix | Laplacian CG and preconditioner tests | already has `build_laplacian_2d`; no Day 8 change needed |
| identity matrix | zero-residual and exact-solution paths | already has `build_identity`; no Day 8 change needed |
| SuiteSparse load plus known RHS setup | `test_cg_nos4`, `test_cg_bcsstk04`, GMRES SuiteSparse comparisons | defer because data-set-specific tolerances and solver claims should remain obvious |
| diagonal matrix-free operator setup | `test_cg_mf_diagonal`, scalar alias test | defer because the inline diagonal-to-solution relation documents the expected result |

### Safe RHS Builders

These candidates are safe only when the helper name is literal and the call
site still shows the solver comparison:

- sequential RHS `b[i] = i + 1` in matrix-free comparison tests;
- sequential exact solution plus `compute_rhs` in CG/GMRES tests, deferred
  because it appears in many convergence-sensitive contexts;
- sine/cosine exact solution plus `compute_rhs`, deferred because each formula
  is part of the conditioning or oracle story.

### Inline Proof Logic

The following logic must remain inline in Day 8:

- all solver option structs, including tolerance, restart, maximum iteration,
  verbosity, and preconditioner side;
- initial guess setup and exact-initial-guess comparisons;
- diagonal, ILU, and right-preconditioner setup and teardown;
- restart loops and restart-comparison arrays;
- convergence, non-convergence, residual, and iteration-count assertions;
- direct comparison assertions between CG and Cholesky, GMRES and LU, and
  matrix-free and concrete solver paths;
- fixed small analytical examples where inline matrix and RHS values explain
  the known solution.

## Selected Day 8 Batch

Limit Day 8 to matrix/RHS setup in the matrix-free comparison block.

### New Helper

Add one local helper near the existing iterative test helpers:

```c
static void fill_sequential_rhs(double *b, idx_t n);
```

The helper should only fill `b[i] = (double)(i + 1)`. It must not allocate
memory, create solver options, run a solver, compute residuals, or assert
convergence.

### Builder Reuse

Reuse existing local builders in the two manually constructed matrix-free basic
tests:

| call site | selected replacement |
|---|---|
| `test_cg_mf_basic` | `build_spd_tridiag(n, 4.0, -1.0)` |
| `test_gmres_mf_basic` | `build_unsym_tridiag(n, 5.0, -2.0, -1.0)` |

The `build_unsym_tridiag` argument order is `diag, upper, lower`, so the GMRES
matrix-free basic test should pass `-2.0` for the upper diagonal and `-1.0` for
the lower diagonal to preserve the current fixture.

### RHS Fill Call Sites

Use `fill_sequential_rhs` only in the matrix-free tests that currently repeat
the same literal RHS loop:

| call site | proof preserved inline |
|---|---|
| `test_cg_mf_basic` | CG and CG matrix-free convergence, iteration count, and solution equality |
| `test_cg_mf_nos4` | SuiteSparse load, CG and CG matrix-free comparison, and max-difference assertion |
| `test_gmres_mf_basic` | GMRES and GMRES matrix-free convergence, iteration count, and solution equality |
| `test_gmres_mf_right_precond` | ILU factorization, right-preconditioned GMRES comparison, and teardown |

Do not broaden Day 8 into exact-solution allocation helpers or preconditioner
setup helpers.

## Placement and Naming

- File: `tests/test_iterative.c`
- Placement: helper near `compute_rhs`, before the first CG test section
- Linkage: `static`
- Return convention: `void`, because allocation and failure handling stay at
  each test site
- Build-system impact: none

No public header, internal header, Makefile, CMake, or `RUN_TEST` update is
needed.

## Validation Plan

Because Day 8 will edit a `.c` file, required validation is:

```sh
make build/test_iterative && ./build/test_iterative
make format && make lint && make test
git diff --check
```

No CTest registration count check is expected because Day 8 should not add,
remove, or rename `RUN_TEST` entries.

## Deferred Iterative Cleanup

The following cleanup remains intentionally deferred:

- exact-solution and RHS allocation helpers for repeated CG/GMRES known
  solution tests;
- SuiteSparse load plus exact-RHS builders;
- restart comparison setup helpers;
- preconditioner setup, diagonal inverse, and ILU context helpers;
- diagonal matrix-free operator builders;
- any helper that includes convergence, residual, iteration-count, or
  direct-solver comparison assertions.

These are candidates for later narrow cleanup batches after the matrix-free
builder/RHS extraction proves the local-helper approach.
