# Sprint 110 Day 10: Iterative CG Proof-Owner Cleanup

## Purpose

Day 10 implemented the Day 9-selected proof-owner cleanup family:
`tests/test_iterative.c` CG exact-RHS allocation/setup. The cleanup was kept
local to the iterative test owner and did not introduce a shared helper header,
compiled helper target, public API, source-list change, or CTest registration
change.

## Implemented Cleanup

- Added a local `iter_exact_rhs_value_fn` callback type in
  `tests/test_iterative.c`.
- Added local sequential and sine-scaled exact-solution generators.
- Added `make_iterative_exact_rhs`, which:
  - allocates `x_exact`;
  - allocates `b`;
  - fills `x_exact` from the caller-selected generator;
  - computes `b = A*x_exact` through the existing `compute_rhs` helper;
  - returns both buffers to the caller.
- Added `require_iterative_exact_rhs` so allocation/setup failures become a
  test failure and return before null buffers can be dereferenced.

## Updated Call Sites

The helper replaced repeated dynamic exact-RHS setup in the following CG tests:

- `test_cg_laplacian_2d`;
- `test_cg_initial_guess`;
- `test_cg_large_tridiag`;
- `test_cg_max_iter_exceeded`;
- `test_cg_nos4`;
- `test_cg_bcsstk04`;
- `test_cg_suitesparse_initial_guess`;
- `test_cg_tight_tolerance`;
- `test_cg_loose_tolerance`;
- `test_cg_residual_accuracy`.

## Proof Visibility Preserved

The cleanup intentionally leaves these proof elements visible at the call
sites:

- every `sparse_solve_cg` call;
- every `sparse_iter_opts_t` value;
- convergence and non-convergence assertions;
- residual thresholds;
- independent residual recomputation;
- iteration comparisons;
- printed residual and iteration labels;
- exact initial-guess setup;
- cleanup ownership.

## Explicit Non-Changes

- No preconditioner-specific CG setup was moved.
- No stack/literal CG proof vectors were moved.
- No GMRES, BiCGSTAB, MINRES, or handle-helper setup was moved.
- No public header, internal header, build-system, source-list, or CTest
  registration changed.

## Validation Plan

Because Day 10 modifies a `.c` test file, required validation is:

- `make build/test_iterative`;
- `build/test_iterative`;
- `make format && make lint && make test`;
- `git diff --check`.

Validation results are recorded in `WORKING_NOTES.md`.

## Residual Follow-Through

Remaining proof-owner cleanup candidates stay deferred for later bounded work:

- CG preconditioner-specific exact-RHS setup;
- GMRES, BiCGSTAB, and MINRES exact-RHS setup families;
- LDLT CSC external dense-reference oracle cleanup;
- QR sequential RHS setup where literals still explain least-squares or
  refinement proof behavior.
